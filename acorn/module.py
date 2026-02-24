"""Core module class for Acorn."""

import copy
import json
import inspect
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import litellm

from acorn.exceptions import AcornError, BranchError, ParseError, ToolConflictError
from acorn.serialization import pydantic_to_xml
from acorn.tool_schema import generate_tool_schema
from acorn.llm import call_llm
from acorn.types import Step, ToolCall, ToolResult


def _resolve_refs(schema: dict, defs: dict) -> dict:
    """Recursively replace $ref pointers with the actual schema from $defs."""
    if isinstance(schema, dict):
        if "$ref" in schema:
            ref_name = schema["$ref"].split("/")[-1]
            resolved = copy.deepcopy(defs[ref_name])
            # The resolved schema may itself contain $refs
            return _resolve_refs(resolved, defs)
        return {k: _resolve_refs(v, defs) for k, v in schema.items()}
    if isinstance(schema, list):
        return [_resolve_refs(item, defs) for item in schema]
    return schema


def _clean_schema(schema: dict) -> None:
    """Recursively remove 'title' keys that Pydantic adds."""
    if isinstance(schema, dict):
        schema.pop("title", None)
        for v in schema.values():
            _clean_schema(v)
    elif isinstance(schema, list):
        for item in schema:
            _clean_schema(item)


# Models for branch listing
class BranchFieldInfo(BaseModel):
    """Schema field information for a branch input."""
    name: str
    required: bool
    description: str = ""


class BranchInfo(BaseModel):
    """Information about an available branch."""
    name: str
    description: str
    input_schema: list[BranchFieldInfo] = []


class AvailableBranches(BaseModel):
    """List of available branches."""
    branches: list[BranchInfo]


class Module:
    """Base class for LLM modules with structured I/O.

    Class attributes (override in subclass):
        model: Model identifier (string) or config dict
               String: model name (e.g., "anthropic/claude-sonnet-4-5-20250514")
               Dict: {
                   "id": "model-name",  # required
                   "vertex_location": "...",  # optional
                   "vertex_credentials": "...",  # optional
                   "reasoning": True | "low" | "medium" | "high"  # optional
               }
        temperature: Sampling temperature (0.0-1.0)
        max_tokens: Maximum tokens to generate
        max_steps: Maximum agentic steps (None = single-turn)

        system_prompt: System prompt (str, Path, or method)
        initial_input: Pydantic model for input schema
        final_output: Pydantic model for output schema (required for single-turn, optional for multi-turn)
        tools: List of tool functions

    Example:
        >>> from pydantic import BaseModel
        >>> class Summarizer(Module):
        ...     class Input(BaseModel):
        ...         text: str
        ...     class Output(BaseModel):
        ...         summary: str
        ...     initial_input = Input
        ...     final_output = Output
        >>> summarizer = Summarizer()
        >>> result = summarizer(text="Long text...")

    Example (multi-turn without final_output):
        >>> class ToolExecutor(Module):
        ...     max_steps = 5
        ...     tools = [log_action, save_data]
        ...     final_output = None  # No structured output
        >>> executor = ToolExecutor()
        >>> result = executor()  # Returns None after executing tools
        >>> assert result is None
    """

    # Default configuration
    model: str | dict = ""
    temperature: float = 0.7
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    top_p: float | None = None
    stop: str | list[str] | None = None
    presence_penalty: float | None = None
    stream_options: dict | None = None
    max_steps: int | None = None  # None = single-turn mode

    # Prompt and schema
    system_prompt: "str | Path | Template" = ""
    initial_input: type[BaseModel] | None = None
    final_output: type[BaseModel] | None = None

    # Tools
    tools: list = []

    # Branches (sub-agent modules)
    branches: list = []

    # Metadata for LiteLLM tracking
    metadata: dict | None = None

    # Model fallbacks for automatic failover
    model_fallbacks: list = []

    # Provider caching configuration
    cache: bool | list[dict] | None = None

    # XML configuration
    xml_input_root: str = "input"
    xml_output_root: str = "output"

    # Parse retry configuration
    max_parse_retries: int = 2

    # Streaming configuration
    stream: bool = False  # Enable streaming (requires on_stream callback)

    def __init__(self):
        """Initialize the module."""
        # Ensure model is set
        if not self.model:
            raise ValueError(
                "model must be set. Provide a model string (e.g., 'anthropic/claude-sonnet-4-5-20250514') "
                "or a model config dict with an 'id' key."
            )

        # Validate model configuration
        self._validate_model_config()

        # Validate model fallbacks
        self._validate_model_fallbacks()

        # Validate cache configuration
        self._validate_cache_config()

        # Validate branches configuration (basic type checks)
        self._validate_branches()

        # Validate final_output requirement for single-turn
        if self.max_steps is None and self.final_output is None:
            raise ValueError(
                "final_output must be defined for single-turn modules (max_steps=None). "
                "Set max_steps to enable multi-turn mode without final_output."
            )

        # Collect all tools (includes branch tool if branches configured)
        self._collected_tools = self._collect_all_tools()

        # Check for tool name conflicts
        self._check_tool_conflicts()

        # History for multi-turn (will be used in Phase 6)
        self.history = []

        # Internal attributes for branching (set during branch execution)
        self._parent_tools = []  # Parent's tools (for call_parent_tool())
        self._inherited_history = None  # Deep-copied parent history

    def _validate_model_config(self):
        """Validate model configuration.

        Raises:
            ValueError: If model config is invalid
        """
        if isinstance(self.model, dict):
            # Validate required keys
            if "id" not in self.model:
                raise ValueError("Model dict must contain 'id' key")

            # Validate allowed keys
            allowed_keys = {"id", "vertex_location", "vertex_credentials", "reasoning", "api_key", "api_base"}
            invalid_keys = set(self.model.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(f"Invalid model config keys: {invalid_keys}. Allowed: {allowed_keys}")

            # Validate reasoning parameter
            if "reasoning" in self.model:
                reasoning = self.model["reasoning"]
                if reasoning is not True and reasoning not in ["low", "medium", "high"]:
                    raise ValueError(
                        f"reasoning must be True or one of 'low', 'medium', 'high', got: {reasoning}"
                    )
                if not litellm.supports_reasoning(model=self.model["id"]):
                    raise ValueError(
                        f"Model '{self.model['id']}' does not support reasoning. "
                        f"Remove the 'reasoning' parameter or use a model that supports reasoning."
                    )

    def _validate_model_fallbacks(self):
        """Validate model_fallbacks configuration.

        Each entry follows the same rules as `model`: string or dict with
        {id, vertex_location, vertex_credentials, reasoning, api_key, api_base}.

        Raises:
            ValueError: If any fallback entry is invalid
        """
        if not self.model_fallbacks:
            return

        allowed_keys = {"id", "vertex_location", "vertex_credentials", "reasoning", "api_key", "api_base"}

        for i, fallback in enumerate(self.model_fallbacks):
            if isinstance(fallback, str):
                continue

            if not isinstance(fallback, dict):
                raise ValueError(
                    f"model_fallbacks[{i}] must be a string or dict, got: {type(fallback).__name__}"
                )

            if "id" not in fallback:
                raise ValueError(f"model_fallbacks[{i}] dict must contain 'id' key")

            invalid_keys = set(fallback.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(
                    f"Invalid model_fallbacks[{i}] config keys: {invalid_keys}. Allowed: {allowed_keys}"
                )

            if "reasoning" in fallback:
                reasoning = fallback["reasoning"]
                if reasoning is not True and reasoning not in ["low", "medium", "high"]:
                    raise ValueError(
                        f"model_fallbacks[{i}] reasoning must be True or one of 'low', 'medium', 'high', got: {reasoning}"
                    )

    def _validate_cache_config(self):
        """Validate cache configuration.

        Raises:
            ValueError: If cache config is invalid
        """
        # Allow None, True, False
        if self.cache is None or isinstance(self.cache, bool):
            return

        # Must be a list
        if not isinstance(self.cache, list):
            raise ValueError(
                f"cache must be None, bool, or list[dict], got: {type(self.cache).__name__}"
            )

        # Must not be empty
        if len(self.cache) == 0:
            raise ValueError("cache list cannot be empty")

        # Validate each item
        for i, item in enumerate(self.cache):
            # Must be a dict
            if not isinstance(item, dict):
                raise ValueError(
                    f"cache[{i}] must be a dict, got: {type(item).__name__}"
                )

            # Must have 'location' key
            if "location" not in item:
                raise ValueError(
                    f"cache[{i}] must have 'location' key"
                )

            # 'location' must be 'message'
            if item["location"] != "message":
                raise ValueError(
                    f"cache[{i}]['location'] must be 'message', got: {item['location']}"
                )

            # Only allowed keys: location, role, index
            allowed_keys = {"location", "role", "index"}
            invalid_keys = set(item.keys()) - allowed_keys
            if invalid_keys:
                raise ValueError(
                    f"cache[{i}] has invalid keys: {invalid_keys}. Allowed: {allowed_keys}"
                )

            # Validate 'role' if present
            if "role" in item and not isinstance(item["role"], str):
                raise ValueError(
                    f"cache[{i}]['role'] must be a string, got: {type(item['role']).__name__}"
                )

            # Validate 'index' if present
            if "index" in item and not isinstance(item["index"], int):
                raise ValueError(
                    f"cache[{i}]['index'] must be an int, got: {type(item['index']).__name__}"
                )

    def _validate_branches(self):
        """Validate branches configuration (type checks only).

        Raises:
            ValueError: If branches config is invalid
        """
        if not self.branches:
            return

        if not isinstance(self.branches, list):
            raise ValueError(
                f"branches must be a list, got: {type(self.branches).__name__}"
            )

        # Check each item is a Module subclass
        seen_names = set()
        for i, branch_class in enumerate(self.branches):
            if not (isinstance(branch_class, type) and issubclass(branch_class, Module)):
                raise ValueError(
                    f"branches[{i}] must be a Module subclass, got: {branch_class}"
                )

            # Check for duplicate class names
            name = branch_class.__name__
            if name in seen_names:
                raise ValueError(f"Duplicate branch class name: {name}")
            seen_names.add(name)

    def __call__(self, **kwargs) -> BaseModel | None:
        """Execute the module with provided inputs.

        Args:
            **kwargs: Input fields matching initial_input schema

        Returns:
            Instance of final_output model, or None if no final_output defined

        Raises:
            AcornError: If execution fails
            ParseError: If output validation fails
        """
        if self.max_steps is None:
            return self._single_turn(**kwargs)
        else:
            return self._agentic_loop(**kwargs)

    def _single_turn(self, **kwargs) -> BaseModel | None:
        """Execute a single-turn module call.

        Args:
            **kwargs: Input fields

        Returns:
            Validated output model (always defined due to __init__ validation)
        """
        # 1. Validate input
        if self.initial_input:
            input_model = self.initial_input(**kwargs)
        else:
            input_model = None

        # 2. Build system message
        system_message = self._build_system_message()

        # 3. Build user message with XML input
        if input_model:
            input_xml = pydantic_to_xml(
                input_model,
                root_tag=self.xml_input_root,
                include_descriptions=True
            )
            user_message = {"role": "user", "content": input_xml}
        else:
            # No initial_input schema
            user_message = {"role": "user", "content": str(kwargs)}

        messages = [system_message]

        # Prepend inherited history if this is a branch
        if self._inherited_history:
            messages.extend(self._inherited_history)

        messages.append(user_message)

        # Initialize history
        self.history = messages.copy()

        # 4. Collect tools and add __finish__
        tools_list = self._collected_tools.copy()

        # Add call_parent_tool if this is a branch with parent tools
        if self._parent_tools:
            tools_list.append(self._generate_parent_tool(self._parent_tools))

        finish_tool = self._generate_finish_tool()
        tools_list.append(finish_tool)

        # Generate tool schemas
        tool_schemas = [self._get_tool_schema(t) for t in tools_list]

        # 5. Call LLM
        on_stream_callback = self.on_stream if (self.stream and hasattr(self, 'on_stream')) else None
        response = call_llm(
            messages=messages,
            model=self.model,
            tools=tool_schemas,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            top_p=self.top_p,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            stream_options=self.stream_options,
            stream=self.stream,
            on_stream=on_stream_callback,
            final_output_schema=self.final_output,
            metadata=self.metadata,
            cache=self.cache,
            model_fallbacks=self.model_fallbacks or None,
        )

        # Add assistant response to history
        assistant_message = {
            "role": "assistant",
            "content": response.get("content"),
        }
        if response.get("reasoning_content"):
            assistant_message["reasoning_content"] = response["reasoning_content"]
        if response.get("tool_calls"):
            assistant_message["tool_calls"] = response["tool_calls"]
        self.history.append(assistant_message)

        # 6. Handle response - retry if no tool calls
        no_tool_call_retries = 0
        while not response.get("tool_calls"):
            no_tool_call_retries += 1
            if no_tool_call_retries > self.max_parse_retries:
                raise AcornError(
                    f"No tool called in single-turn mode after {self.max_parse_retries} retries"
                )
            self.history.append({
                "role": "user",
                "content": (
                    "You must respond by calling one of the available tools. "
                    "Use the provided tools to take action, or call __finish__ "
                    "with the appropriate arguments to complete the task."
                )
            })
            response = call_llm(
                messages=self.history,
                model=self.model,
                tools=tool_schemas,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_completion_tokens=self.max_completion_tokens,
                top_p=self.top_p,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                stream_options=self.stream_options,
                stream=self.stream,
                on_stream=on_stream_callback,
                final_output_schema=self.final_output,
                metadata=self.metadata,
                cache=self.cache,
                model_fallbacks=self.model_fallbacks or None,
            )
            assistant_message = {
                "role": "assistant",
                "content": response.get("content"),
            }
            if response.get("reasoning_content"):
                assistant_message["reasoning_content"] = response["reasoning_content"]
            if response.get("tool_calls"):
                assistant_message["tool_calls"] = response["tool_calls"]
            self.history.append(assistant_message)

        tool_call = response["tool_calls"][0]

        if tool_call["function"]["name"] != "__finish__":
            raise AcornError(
                f"Non-finish tool called in single-turn mode: {tool_call['function']['name']}"
            )

        # 7. Validate __finish__ arguments (with retries)
        return self._validate_and_retry(
            messages,
            tool_schemas,
            tool_call,
            attempt=0
        )

    def _agentic_loop(self, **kwargs) -> BaseModel | None:
        """Execute multi-turn agentic loop with tool calls.

        Args:
            **kwargs: Input fields

        Returns:
            Validated output model, or None if no final_output defined
        """
        # 1. Validate input
        if self.initial_input:
            input_model = self.initial_input(**kwargs)
        else:
            input_model = None

        # 2. Build system message
        system_message = self._build_system_message()

        # 3. Build initial user message
        if input_model:
            input_xml = pydantic_to_xml(
                input_model,
                root_tag=self.xml_input_root,
                include_descriptions=True
            )
            user_message = {"role": "user", "content": input_xml}
        else:
            user_message = {"role": "user", "content": str(kwargs)}

        # Initialize history
        self.history = [system_message]

        # Prepend inherited history if this is a branch
        if self._inherited_history:
            self.history.extend(self._inherited_history)

        self.history.append(user_message)

        # 4. Collect tools and add __finish__
        current_tools = self._collected_tools.copy()

        # Add call_parent_tool if this is a branch with parent tools
        if self._parent_tools:
            current_tools.append(self._generate_parent_tool(self._parent_tools))

        finish_tool = self._generate_finish_tool()
        current_tools.append(finish_tool)

        # Loop state
        step_count = 0
        no_tool_call_retries = 0

        while step_count < self.max_steps:
            step_count += 1

            # Generate tool schemas
            tool_schemas = [self._get_tool_schema(t) for t in current_tools]

            # Call LLM
            on_stream_callback = self.on_stream if (self.stream and hasattr(self, 'on_stream')) else None
            response = call_llm(
                messages=self.history,
                model=self.model,
                tools=tool_schemas,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_completion_tokens=self.max_completion_tokens,
                top_p=self.top_p,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                stream_options=self.stream_options,
                stream=self.stream,
                on_stream=on_stream_callback,
                final_output_schema=self.final_output,
                metadata=self.metadata,
                cache=self.cache,
                model_fallbacks=self.model_fallbacks or None,
            )

            # Add assistant message to history
            assistant_message = {
                "role": "assistant",
                "content": response.get("content"),
            }

            # Add reasoning_content if present
            if response.get("reasoning_content"):
                assistant_message["reasoning_content"] = response["reasoning_content"]

            # Add tool calls to assistant message if present
            if response.get("tool_calls"):
                assistant_message["tool_calls"] = response["tool_calls"]

            self.history.append(assistant_message)

            # Get tool calls
            tool_calls = response.get("tool_calls", [])

            if not tool_calls:
                no_tool_call_retries += 1
                if no_tool_call_retries > self.max_parse_retries:
                    raise AcornError(
                        f"No tool calls in agentic loop step after {self.max_parse_retries} retries"
                    )
                self.history.append({
                    "role": "user",
                    "content": (
                        "You must respond by calling one of the available tools. "
                        "Use the provided tools to take action, or call __finish__ "
                        "with the appropriate arguments to complete the task."
                    )
                })
                continue

            # Process tool calls
            tool_call_objs = []
            tool_result_objs = []

            for tc in tool_calls:
                tool_name = tc["function"]["name"]

                # Check if it's __finish__
                if tool_name == "__finish__":
                    # If no final_output, return None immediately
                    if self.final_output is None:
                        return None

                    # Validate and return
                    try:
                        arguments = json.loads(tc["function"]["arguments"])
                        result = self.final_output(**arguments)
                        return result
                    except Exception as e:
                        # Validation failed - add error and continue loop
                        error_msg = {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": f"Error: Output validation failed: {e}\nPlease call __finish__ again with valid arguments."
                        }
                        self.history.append(error_msg)
                        continue

                # Execute regular tool
                tool_call_obj = ToolCall(
                    id=tc["id"],
                    name=tool_name,
                    arguments=json.loads(tc["function"]["arguments"])
                )
                tool_call_objs.append(tool_call_obj)

                # Find and execute tool
                tool_result = self._execute_tool(tool_call_obj, current_tools)
                tool_result_objs.append(tool_result)

                # Add tool result to history
                result_msg = {
                    "role": "tool",
                    "tool_call_id": tool_result.id,
                    "content": str(tool_result.output) if tool_result.error is None else f"Error: {tool_result.error}"
                }
                self.history.append(result_msg)

            # Reset no-tool-call retry counter after successful tool processing
            no_tool_call_retries = 0

            # Build Step object
            step = Step(
                counter=step_count,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                tools=current_tools.copy(),
                response=response,
                tool_calls=tool_call_objs,
                tool_results=tool_result_objs,
            )

            # Call on_step callback if defined
            if hasattr(self, 'on_step') and callable(self.on_step):
                step = self.on_step(step)

                # Check if step.finish() was called
                if step._finished:
                    # If no final_output, ignore kwargs and return None
                    if self.final_output is None:
                        return None

                    try:
                        result = self.final_output(**step._finish_kwargs)
                        return result
                    except Exception as e:
                        raise ParseError(
                            f"Failed to validate output from step.finish(): {e}",
                            raw_output=step._finish_kwargs
                        )

                # Apply step mutations
                for tool_to_add in step._tools_to_add:
                    if tool_to_add not in current_tools:
                        current_tools.append(tool_to_add)

                for tool_name_to_remove in step._tools_to_remove:
                    current_tools = [t for t in current_tools if t.__name__ != tool_name_to_remove]

        # Max steps reached - force termination with __finish__
        return self._force_termination()

    def _execute_tool(self, tool_call: ToolCall, tools: list) -> ToolResult:
        """Execute a tool and return the result.

        Args:
            tool_call: ToolCall object with tool name and arguments
            tools: List of available tools

        Returns:
            ToolResult object
        """
        # Find the tool
        tool_func = None
        for t in tools:
            if t.__name__ == tool_call.name:
                tool_func = t
                break

        if tool_func is None:
            return ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=None,
                error=f"Tool '{tool_call.name}' not found"
            )

        # Execute the tool
        try:
            result = tool_func(**tool_call.arguments)
            return ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=result,
                error=None
            )
        except Exception as e:
            return ToolResult(
                id=tool_call.id,
                name=tool_call.name,
                output=None,
                error=str(e)
            )

    def _build_system_message(self) -> dict:
        """Build the system message from system_prompt.

        Returns:
            System message dictionary
        """
        # Determine system prompt source
        from acorn.template import Template

        if isinstance(self.system_prompt, Template):
            prompt_text = self.system_prompt.render()
        elif isinstance(self.system_prompt, Path):
            # Resolve relative paths against the defining class's source file
            path = self.system_prompt
            if not path.is_absolute():
                import inspect
                class_file = Path(inspect.getfile(self.__class__)).resolve().parent
                path = class_file / path
            prompt_text = path.read_text()
        elif isinstance(self.system_prompt, str) and self.system_prompt:
            # Use string directly
            prompt_text = self.system_prompt
        elif callable(getattr(self.__class__, 'system_prompt', None)):
            # Call class method
            prompt_text = self.system_prompt()
        else:
            # No system prompt
            prompt_text = ""

        return {"role": "system", "content": prompt_text}

    def _validate_and_retry(
        self,
        messages: list[dict],
        tool_schemas: list[dict],
        tool_call: dict,
        attempt: int
    ) -> BaseModel:
        """Validate __finish__ output with retry logic.

        Args:
            messages: Message history
            tool_schemas: Available tool schemas
            tool_call: The __finish__ tool call to validate
            attempt: Current retry attempt number

        Returns:
            Validated output model

        Raises:
            ParseError: If validation fails after all retries
        """
        try:
            arguments = json.loads(tool_call["function"]["arguments"])
            result = self.final_output(**arguments)
            return result
        except Exception as e:
            # Validation failed
            if attempt >= self.max_parse_retries:
                # Out of retries
                raise ParseError(
                    f"Failed to validate output after {attempt} retries: {e}",
                    raw_output=tool_call["function"]["arguments"]
                )

            # Add error message and retry
            error_msg = {
                "role": "user",
                "content": f"Error: Output validation failed: {e}\n\nPlease call __finish__ again with valid arguments matching the schema."
            }
            retry_messages = messages + [
                {"role": "assistant", "content": tool_call["function"]["name"]},
                error_msg
            ]

            # Retry LLM call
            response = call_llm(
                messages=retry_messages,
                model=self.model,
                tools=tool_schemas,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_completion_tokens=self.max_completion_tokens,
                top_p=self.top_p,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                stream_options=self.stream_options,
                metadata=self.metadata,
                cache=self.cache,
                model_fallbacks=self.model_fallbacks or None,
            )

            if not response.get("tool_calls"):
                raise ParseError(
                    f"No tool called in retry attempt {attempt + 1}",
                    raw_output=None
                )

            retry_tool_call = response["tool_calls"][0]

            if retry_tool_call["function"]["name"] != "__finish__":
                raise ParseError(
                    f"Wrong tool called in retry: {retry_tool_call['function']['name']}",
                    raw_output=None
                )

            # Recursive retry
            return self._validate_and_retry(
                retry_messages,
                tool_schemas,
                retry_tool_call,
                attempt + 1
            )

    def _collect_all_tools(self) -> list:
        """Collect tools from tools attribute and @tool decorated methods.

        Returns:
            List of tool functions
        """
        collected = []

        # Add from tools attribute
        collected.extend(self.tools)

        # Add from @tool decorated methods
        for name in dir(self):
            # Skip private/magic methods
            if name.startswith("_"):
                continue

            attr = getattr(self, name)

            # Check if it's a tool-decorated function/method
            if callable(attr) and hasattr(attr, "_tool_schema"):
                collected.append(attr)

        # Add branch tool if branches are configured
        if self.branches:
            collected.append(self._generate_branch_tool())

        return collected

    def _check_tool_conflicts(self):
        """Check for tool name conflicts.

        Raises:
            ToolConflictError: If duplicate tool names found
        """
        names = set()

        for tool in self._collected_tools:
            name = tool.__name__
            if name in names:
                raise ToolConflictError(f"Duplicate tool name: {name}")
            names.add(name)

    def _generate_branch_tool(self):
        """Generate the branch() tool for spawning sub-agent branches.

        Returns:
            A callable with _tool_schema for branch invocation.
        """
        module_ref = self

        def branch(**kwargs):
            """Spawn a sub-agent branch. IMPORTANT: You MUST first call branch() with no arguments to discover available branches and their exact input parameter names. Do NOT guess parameter names. After discovery, call with 'name' and the branch's required input parameters."""
            name = kwargs.pop("name", None)
            merge = kwargs.pop("merge", "end_result")

            if name is None:
                # List mode: return available branches in XML
                branch_list = []
                for bclass in module_ref.branches:
                    # Build input schema field list
                    fields = []
                    if bclass.initial_input:
                        for field_name, field_info in bclass.initial_input.model_fields.items():
                            fields.append(BranchFieldInfo(
                                name=field_name,
                                required=field_info.is_required(),
                                description=field_info.description or ""
                            ))

                    branch_list.append(BranchInfo(
                        name=bclass.__name__,
                        description=bclass.__doc__ or "",
                        input_schema=fields
                    ))

                available = AvailableBranches(branches=branch_list)
                return pydantic_to_xml(available, root_tag="available_branches", include_descriptions=False)

            # Execution mode - find branch class by name
            branch_class = None
            for bclass in module_ref.branches:
                if bclass.__name__ == name:
                    branch_class = bclass
                    break

            if branch_class is None:
                available = [b.__name__ for b in module_ref.branches]
                raise BranchError(f"Branch '{name}' not found. Available: {available}")

            if merge not in ("end_result", "summarize"):
                raise BranchError(f"Invalid merge strategy '{merge}'. Must be: end_result, summarize")

            result, merged_content = module_ref._execute_branch(branch_class, merge, **kwargs)
            return merged_content

        branch.__name__ = "branch"

        branch._tool_schema = {
            "type": "function",
            "function": {
                "name": "branch",
                "description": (
                    "Spawn a sub-agent branch. IMPORTANT: You MUST first call branch() with no arguments "
                    "to discover available branches and their exact input parameter names. Do NOT guess "
                    "parameter names. After discovery, call with 'name' set to the branch name and the "
                    "branch's required input parameters using the exact names from the discovery response."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the branch to spawn"},
                        "merge": {
                            "type": "string",
                            "enum": ["summarize", "end_result"],
                            "description": "How to merge branch results. Default: end_result"
                        }
                    },
                    "additionalProperties": True
                }
            }
        }

        return branch

    def _generate_parent_tool(self, parent_tools):
        """Generate the call_parent_tool() tool for accessing parent module's tools.

        Args:
            parent_tools: List of parent's collected tools

        Returns:
            A callable with _tool_schema for parent tool invocation.
        """
        # Filter out __finish__ and branch from parent tools
        filtered_tools = [
            t for t in parent_tools
            if t.__name__ not in ("__finish__", "branch")
        ]

        def call_parent_tool(**kwargs):
            """Call a tool from the parent module. Call with no args to list available parent tools. Call with name and the tool's args to execute it."""
            name = kwargs.pop("name", None)

            if name is None:
                # List mode
                tool_list = []
                for t in filtered_tools:
                    schema = generate_tool_schema(t) if not hasattr(t, "_tool_schema") else t._tool_schema
                    tool_list.append({
                        "name": t.__name__,
                        "schema": schema.get("function", schema)
                    })
                return json.dumps(tool_list, indent=2)

            # Execution mode
            tool_func = None
            for t in filtered_tools:
                if t.__name__ == name:
                    tool_func = t
                    break

            if tool_func is None:
                available = [t.__name__ for t in filtered_tools]
                raise BranchError(f"Parent tool '{name}' not found. Available: {available}")

            return tool_func(**kwargs)

        call_parent_tool.__name__ = "call_parent_tool"

        call_parent_tool._tool_schema = {
            "type": "function",
            "function": {
                "name": "call_parent_tool",
                "description": (
                    "Call a tool from the parent module. Call with no args to list available parent tools. "
                    "Call with name and the tool's args to execute it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Name of the parent tool to call"}
                    },
                    "additionalProperties": True
                }
            }
        }

        return call_parent_tool

    def _execute_branch(self, branch_class, merge="end_result", **kwargs):
        """Execute a branch module and return result with merged content.

        Args:
            branch_class: The Module subclass to execute as a branch
            merge: Merge strategy ('end_result', 'summarize')
            **kwargs: Arguments passed to the branch module's initial_input

        Returns:
            Tuple of (result BaseModel instance or None, merged_content string)

        Raises:
            BranchError: If branch execution fails
        """
        # Deep-copy parent history, excluding the current in-progress step.
        # During parallel tool execution, self.history contains the assistant
        # message with tool_use blocks that don't yet all have matching
        # tool_result messages. This is invalid for providers like Anthropic
        # that require every tool_use to be immediately followed by a
        # tool_result. We strip the incomplete step from the branch's copy.
        branch_history = copy.deepcopy(self.history)

        # Find the last assistant message with tool_calls — if it doesn't have
        # a complete set of tool_results after it, remove it and any partial
        # tool_results that follow.
        for i in range(len(branch_history) - 1, -1, -1):
            msg = branch_history[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                expected_ids = {tc["id"] for tc in msg["tool_calls"]}
                actual_ids = {
                    branch_history[j].get("tool_call_id")
                    for j in range(i + 1, len(branch_history))
                    if branch_history[j].get("role") == "tool"
                }
                if expected_ids != actual_ids:
                    # Incomplete step — trim from this assistant message onward
                    branch_history = branch_history[:i]
                break

        # Instantiate branch
        try:
            branch_instance = branch_class()
        except Exception as e:
            raise BranchError(f"Failed to instantiate branch: {e}")

        # Set parent tools for call_parent_tool() access
        branch_instance._parent_tools = self._collected_tools.copy()

        # Set inherited history (parent context minus system message)
        branch_instance._inherited_history = [
            msg for msg in branch_history if msg.get("role") != "system"
        ]

        # If parent has cache=True, override branch cache to mark inherited
        # history as cacheable. This avoids re-processing the parent context
        # on every LLM call within the branch.
        if self.cache is True and branch_instance._inherited_history:
            # Cache breakpoints: system message + last inherited history message.
            # Messages layout: [system, *inherited_history, branch_user_msg, ...]
            # The last inherited message is at index len(inherited_history).
            last_inherited_idx = len(branch_instance._inherited_history)
            branch_instance.cache = [
                {"location": "message", "role": "system"},
                {"location": "message", "index": last_inherited_idx},
            ]

        # Execute branch
        try:
            result = branch_instance(**kwargs)
        except Exception as e:
            raise BranchError(f"Branch execution failed: {e}")

        # Apply merge strategy
        if result is None and merge == "end_result":
            # Branch has no final_output — fall back to summarize so the
            # parent gets useful context about what the branch did.
            merge = "summarize"

        if merge == "end_result":
            merged_content = self._merge_end_result(result)
        elif merge == "summarize":
            merged_content = self._merge_summarize(branch_instance, result)
        else:
            merged_content = self._merge_end_result(result)

        return (result, merged_content)

    def _merge_end_result(self, result):
        """Merge strategy: return only the serialized final_output.

        Args:
            result: The branch's final_output instance (or None)

        Returns:
            XML string of the result
        """
        if result is None:
            # Create a simple model for None result
            class BranchResult(BaseModel):
                status: str = "completed"
                result: str = "None"

            return pydantic_to_xml(BranchResult(), root_tag="branch_result", include_descriptions=False)

        return pydantic_to_xml(result, root_tag="branch_result", include_descriptions=False)

    def _merge_summarize(self, branch_instance, result):
        """Merge strategy: LLM-generated summary of branch history + result.

        Args:
            branch_instance: The executed branch module instance
            result: The branch's final_output instance (or None)

        Returns:
            Summary string
        """
        # Build summary prompt
        history_text = self._format_branch_history(branch_instance.history)
        result_text = result.model_dump_json(indent=2) if result else "None"

        summary_messages = [
            {"role": "system", "content": "Summarize the following branch execution concisely. Include key findings, actions taken, and the final result."},
            {"role": "user", "content": f"Branch history:\n{history_text}\n\nFinal result:\n{result_text}"}
        ]

        # Use branch's model for summary
        response = call_llm(
            messages=summary_messages,
            model=branch_instance.model,
            tools=None,
            temperature=0.3,
            max_tokens=branch_instance.max_tokens,
        )

        summary = response.get("content", "")
        return f"Branch summary:\n{summary}\n\nFinal result:\n{result_text}"

    def _format_branch_history(self, history):
        """Format branch history as readable text.

        Args:
            history: List of message dicts

        Returns:
            Formatted string
        """
        lines = []
        for msg in history:
            role = msg.get("role", "unknown")
            if role == "system":
                continue  # Skip system message
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])

            if role == "assistant" and tool_calls:
                for tc in tool_calls:
                    func = tc.get("function", tc)
                    name = func.get("name", "unknown") if isinstance(func, dict) else getattr(func, "name", "unknown")
                    args = func.get("arguments", "{}") if isinstance(func, dict) else getattr(func, "arguments", "{}")
                    lines.append(f"[Assistant] Called {name}({args})")
                if content:
                    lines.append(f"[Assistant] {content}")
            elif role == "tool":
                lines.append(f"[Tool Result] {content}")
            elif content:
                lines.append(f"[{role.title()}] {content}")
        return "\n".join(lines)

    def branch(self, module_class, /, merge="end_result", **kwargs):
        """Manually spawn a branch from within on_step or other callbacks.

        Args:
            module_class: The Module subclass to execute (positional-only)
            merge: Merge strategy ('end_result', 'summarize')
            **kwargs: Arguments passed to the branch module's initial_input

        Returns:
            The branch's final_output instance (or None)
        """
        result, merged_content = self._execute_branch(module_class, merge, **kwargs)

        # Inject merge result into parent history
        self.history.append({
            "role": "user",
            "content": f"[Branch Result]\n{merged_content}"
        })

        return result

    def _generate_finish_tool(self) -> dict:
        """Generate the __finish__ tool from final_output schema.

        Returns:
            Tool schema dictionary for __finish__
        """
        # Handle None case - generate parameter-less finish tool
        if not self.final_output:
            finish_schema = {
                "type": "function",
                "function": {
                    "name": "__finish__",
                    "description": "Call this function when you are done executing tools and want to complete the task.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }

            def __finish__(**kwargs):
                return kwargs

            __finish__._tool_schema = finish_schema
            return __finish__

        # Build parameters from final_output model's JSON schema
        schema = self.final_output.model_json_schema()

        # Resolve $defs references inline for nested models
        defs = schema.pop("$defs", {})
        if defs:
            schema = _resolve_refs(schema, defs)

        # Remove Pydantic extras the LLM doesn't need
        _clean_schema(schema)

        # Create __finish__ tool schema
        finish_schema = {
            "type": "function",
            "function": {
                "name": "__finish__",
                "description": "Call this function when you have the final output ready.",
                "parameters": schema
            }
        }

        # Return a callable that has this schema
        def __finish__(**kwargs):
            return kwargs

        __finish__._tool_schema = finish_schema
        return __finish__

    def _get_tool_schema(self, tool: Any) -> dict:
        """Get the schema for a tool.

        Args:
            tool: Tool function or callable

        Returns:
            Tool schema dictionary
        """
        if hasattr(tool, "_tool_schema"):
            return tool._tool_schema
        else:
            # Generate schema on the fly
            return generate_tool_schema(tool)

    def _force_termination(self) -> BaseModel | None:
        """Force termination at max_steps by requiring __finish__ call.

        Primary strategy: Use tool_choice to force __finish__ (preserves cache)
        Fallback strategy: Append XML instruction if tool_choice fails

        Returns:
            Validated output model, or None if no final_output defined

        Raises:
            AcornError: If the model did not call __finish__ within max_steps
        """
        # If no final_output, just return None
        if self.final_output is None:
            return None

        # Collect current tools and generate schemas
        current_tools = self._collected_tools.copy()
        finish_tool = self._generate_finish_tool()
        current_tools.append(finish_tool)
        tool_schemas = [self._get_tool_schema(t) for t in current_tools]

        # Try tool_choice first (preserves prompt caching)
        try:
            response = call_llm(
                messages=self.history,
                model=self.model,
                tools=tool_schemas,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                max_completion_tokens=self.max_completion_tokens,
                top_p=self.top_p,
                stop=self.stop,
                presence_penalty=self.presence_penalty,
                stream_options=self.stream_options,
                tool_choice={"type": "function", "function": {"name": "__finish__"}},
                metadata=self.metadata,
                cache=self.cache,
                model_fallbacks=self.model_fallbacks or None,
            )

            # Process the response
            if response.get("tool_calls"):
                tool_call = response["tool_calls"][0]
                if tool_call["function"]["name"] == "__finish__":
                    # Validate and return (with retry support)
                    try:
                        arguments = json.loads(tool_call["function"]["arguments"])
                        result = self.final_output(**arguments)
                        return result
                    except Exception as e:
                        # Validation failed - use retry mechanism
                        return self._retry_forced_finish(
                            tool_schemas,
                            tool_call,
                            attempt=0,
                            error=e
                        )

        except Exception:
            # tool_choice not supported or failed - fall through to error
            pass

        # Force termination failed — raise a clear error
        raise AcornError(
            f"Module reached max_steps ({self.max_steps}) without calling __finish__. "
            f"The model did not produce a final output in the allowed number of steps. "
            f"Consider increasing max_steps or simplifying the task."
        )

    def _retry_forced_finish(
        self,
        tool_schemas: list[dict],
        tool_call: dict,
        attempt: int,
        error: Exception
    ) -> BaseModel:
        """Retry forced finish after validation failure.

        Args:
            tool_schemas: Available tool schemas
            tool_call: The failed __finish__ tool call
            attempt: Current retry attempt number
            error: The validation error

        Returns:
            Validated output model

        Raises:
            ParseError: If validation fails after all retries
        """
        if attempt >= self.max_parse_retries:
            # Out of retries
            raise ParseError(
                f"Failed to validate forced output after {attempt} retries: {error}",
                raw_output=tool_call["function"]["arguments"]
            )

        # Add error message and retry
        error_msg = {
            "role": "user",
            "content": f"Error: Output validation failed: {error}\n\nPlease call __finish__ again with valid arguments matching the schema."
        }

        # Add assistant message and error to history
        retry_history = self.history + [
            {"role": "assistant", "tool_calls": [tool_call]},
            {"role": "tool", "tool_call_id": tool_call["id"], "content": error_msg["content"]}
        ]

        # Retry with tool_choice
        response = call_llm(
            messages=retry_history,
            model=self.model,
            tools=tool_schemas,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            max_completion_tokens=self.max_completion_tokens,
            top_p=self.top_p,
            stop=self.stop,
            presence_penalty=self.presence_penalty,
            stream_options=self.stream_options,
            tool_choice={"type": "function", "function": {"name": "__finish__"}},
            metadata=self.metadata,
            cache=self.cache,
            model_fallbacks=self.model_fallbacks or None,
        )

        if not response.get("tool_calls"):
            raise ParseError(
                f"No tool called in forced termination retry {attempt + 1}",
                raw_output=None
            )

        retry_tool_call = response["tool_calls"][0]

        if retry_tool_call["function"]["name"] != "__finish__":
            raise ParseError(
                f"Wrong tool called in forced termination retry: {retry_tool_call['function']['name']}",
                raw_output=None
            )

        # Try to validate again
        try:
            arguments = json.loads(retry_tool_call["function"]["arguments"])
            result = self.final_output(**arguments)
            return result
        except Exception as e:
            # Recursive retry
            return self._retry_forced_finish(
                tool_schemas,
                retry_tool_call,
                attempt + 1,
                error=e
            )

