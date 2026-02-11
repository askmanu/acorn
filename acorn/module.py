"""Core module class for Acorn."""

import json
import inspect
from pathlib import Path
from typing import Any
from pydantic import BaseModel
import litellm

from acorn.exceptions import AcornError, ParseError, ToolConflictError
from acorn.serialization import pydantic_to_xml
from acorn.tool_schema import generate_tool_schema
from acorn.llm import call_llm
from acorn.types import Step, ToolCall, ToolResult


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
    model: str | dict = "anthropic/claude-sonnet-4-5-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_steps: int | None = None  # None = single-turn mode

    # Prompt and schema
    system_prompt: str | Path = ""
    initial_input: type[BaseModel] | None = None
    final_output: type[BaseModel] | None = None

    # Tools
    tools: list = []

    # Metadata for LiteLLM tracking
    metadata: dict | None = None

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
        # Validate model configuration
        self._validate_model_config()

        # Validate cache configuration
        self._validate_cache_config()

        # Validate final_output requirement for single-turn
        if self.max_steps is None and self.final_output is None:
            raise ValueError(
                "final_output must be defined for single-turn modules (max_steps=None). "
                "Set max_steps to enable multi-turn mode without final_output."
            )

        # Collect all tools
        self._collected_tools = self._collect_all_tools()

        # Check for tool name conflicts
        self._check_tool_conflicts()

        # History for multi-turn (will be used in Phase 6)
        self.history = []

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

        messages = [system_message, user_message]

        # Initialize history
        self.history = messages.copy()

        # 4. Collect tools and add __finish__
        tools_list = self._collected_tools.copy()
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
            stream=self.stream,
            on_stream=on_stream_callback,
            final_output_schema=self.final_output,
            metadata=self.metadata,
            cache=self.cache,
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
                stream=self.stream,
                on_stream=on_stream_callback,
                final_output_schema=self.final_output,
                metadata=self.metadata,
                cache=self.cache,
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
        self.history = [system_message, user_message]

        # 4. Collect tools and add __finish__
        current_tools = self._collected_tools.copy()
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
                stream=self.stream,
                on_stream=on_stream_callback,
                final_output_schema=self.final_output,
                metadata=self.metadata,
                cache=self.cache,
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
        if isinstance(self.system_prompt, Path):
            # Load from file
            prompt_text = self.system_prompt.read_text()
        elif isinstance(self.system_prompt, str) and self.system_prompt:
            # Use string directly
            prompt_text = self.system_prompt
        elif callable(getattr(self.__class__, 'system_prompt', None)):
            # Call class method
            prompt_text = self.system_prompt()
        elif self.__doc__:
            # Use class docstring
            prompt_text = self.__doc__
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
                metadata=self.metadata,
                cache=self.cache,
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

        # Build parameters from final_output model fields
        parameters = {
            "type": "object",
            "properties": {},
            "required": []
        }

        for field_name, field_info in self.final_output.model_fields.items():
            # Convert field type to JSON schema
            field_schema = {"type": "string"}  # Simplified for now

            # Add description if available
            if field_info.description:
                field_schema["description"] = field_info.description

            parameters["properties"][field_name] = field_schema

            # Add to required if not optional
            if field_info.is_required():
                parameters["required"].append(field_name)

        # Create __finish__ tool schema
        finish_schema = {
            "type": "function",
            "function": {
                "name": "__finish__",
                "description": "Call this function when you have the final output ready.",
                "parameters": parameters
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
            ParseError: If forced output fails validation after all retries
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
                tool_choice={"type": "function", "function": {"name": "__finish__"}},
                metadata=self.metadata,
                cache=self.cache,
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

        except Exception as e:
            # tool_choice not supported or failed - fall back to XML
            return self._force_termination_xml()

        # No tool call or wrong tool - fall back to XML
        return self._force_termination_xml()

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
            tool_choice={"type": "function", "function": {"name": "__finish__"}},
            metadata=self.metadata,
            cache=self.cache,
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

    def _force_termination_xml(self) -> BaseModel:
        """Force termination using XML fallback (when tool_choice not supported).

        Appends XML instruction to history and parses XML response.

        Returns:
            Validated output model

        Raises:
            ParseError: If XML parsing or validation fails
        """
        from acorn.serialization import xml_to_pydantic

        # Build XML template with field descriptions as attributes
        xml_template_lines = [f"<{self.xml_output_root}>"]

        for field_name, field_info in self.final_output.model_fields.items():
            desc_attr = ""
            if field_info.description:
                # Escape quotes in description
                escaped_desc = field_info.description.replace('"', '&quot;')
                desc_attr = f' description="{escaped_desc}"'

            xml_template_lines.append(f"    <{field_name}{desc_attr}></{field_name}>")

        xml_template_lines.append(f"</{self.xml_output_root}>")
        xml_template = "\n".join(xml_template_lines)

        # Append instruction to history
        instruction = (
            f"You must provide your final answer now. "
            f"Respond with your answer in the following XML structure:\n\n"
            f"{xml_template}\n\n"
            f"Fill in the values. Do not repeat the descriptions."
        )

        forced_history = self.history + [
            {"role": "user", "content": instruction}
        ]

        # Call LLM without tools (text-only response)
        response = call_llm(
            messages=forced_history,
            model=self.model,
            tools=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata=self.metadata,
            cache=self.cache,
        )

        # Parse XML from response content
        content = response.get("content", "")

        if not content:
            raise ParseError(
                "Empty response in XML forced termination",
                raw_output=content
            )

        try:
            # Parse XML to Pydantic model
            result = xml_to_pydantic(content, self.final_output)
            return result
        except Exception as e:
            # Try retry mechanism
            return self._retry_xml_forced_finish(
                forced_history,
                content,
                attempt=0,
                error=e
            )

    def _retry_xml_forced_finish(
        self,
        history: list[dict],
        failed_output: str,
        attempt: int,
        error: Exception
    ) -> BaseModel:
        """Retry XML forced finish after parsing/validation failure.

        Args:
            history: Message history with XML instruction
            failed_output: The failed XML output
            attempt: Current retry attempt number
            error: The parsing/validation error

        Returns:
            Validated output model

        Raises:
            ParseError: If validation fails after all retries
        """
        from acorn.serialization import xml_to_pydantic

        if attempt >= self.max_parse_retries:
            # Out of retries
            raise ParseError(
                f"Failed to parse/validate XML forced output after {attempt} retries: {error}",
                raw_output=failed_output
            )

        # Add error message and retry
        error_msg = {
            "role": "user",
            "content": f"Error: Output parsing/validation failed: {error}\n\nPlease provide the XML output again with valid values."
        }

        retry_history = history + [
            {"role": "assistant", "content": failed_output},
            error_msg
        ]

        # Retry LLM call
        response = call_llm(
            messages=retry_history,
            model=self.model,
            tools=None,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            metadata=self.metadata,
            cache=self.cache,
        )

        content = response.get("content", "")

        if not content:
            raise ParseError(
                f"Empty response in XML forced termination retry {attempt + 1}",
                raw_output=content
            )

        try:
            # Parse XML to Pydantic model
            result = xml_to_pydantic(content, self.final_output)
            return result
        except Exception as e:
            # Recursive retry
            return self._retry_xml_forced_finish(
                retry_history,
                content,
                attempt + 1,
                error=e
            )
