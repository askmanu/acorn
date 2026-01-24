# API Reference

Complete API surface for acorn v0.1.

---

## Module

### Class Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model` | `str \| dict` | required | LiteLLM model identifier or config dict |
| `temperature` | `float` | `0.7` | Sampling temperature (0-2) |
| `max_tokens` | `int` | `4096` | Maximum tokens in response |
| `max_steps` | `int \| None` | `None` | Maximum agentic loop iterations. `None` = single-turn (input → output or tool call). Set to `N` for up to N loop iterations. |
| `parse_retries` | `int` | `2` | Retries on output validation failure |
| `system_prompt` | `str \| Path \| method \| docstring` | `""` | System instructions (Markdown). If not set, uses class docstring. |
| `initial_input` | `BaseModel \| list` | `[]` | Input schema |
| `final_output` | `BaseModel \| list` | `[]` | Output schema |
| `tools` | `list[Callable]` | `[]` | Available tools |
| `branches` | `dict` | `{}` | Branch module registry |
| `cache` | `list[dict]` | `[]` | Provider cache configuration |

### XML Serialization Attributes

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `xml_include_descriptions` | `bool` | `True` | Include field descriptions |
| `xml_description_format` | `str` | `"attribute"` | `"attribute"` (description="...") or `"comment"` (<!-- ... -->) |
| `xml_input_root` | `str` | `"input"` | Root element for input |
| `xml_output_root` | `str` | `"output"` | Root element for output |
| `xml_context_root` | `str` | `"context"` | Root element for context |

### Methods

#### `__call__(**kwargs) -> BaseModel`

Run the module with provided inputs.

```python
agent = MyAgent()
result = agent(question="What is Python?")
```

**Parameters**: Must match `initial_input` schema
**Returns**: Instance of `final_output` schema

#### `on_step(step: Step) -> Step`

Callback after each step. Override to customize behavior.

```python
def on_step(self, step):
    print(f"Step {step.counter}")
    return step
```

#### `on_stream(chunk: StreamChunk) -> None`

Callback for streaming responses. Override to handle text chunks and partial structured output.

```python
def on_stream(self, chunk):
    # Handle text content (chain-of-thought)
    if chunk.content:
        print(chunk.content, end="")

    # Handle partial structured output (as __finish__ streams)
    if chunk.partial:
        if chunk.partial.answer:
            print(f"Answer so far: {chunk.partial.answer}")
```

#### `branch(module_class: type) -> BaseModel`

Manually spawn a branch module. The branch inherits current history and context.

```python
result = self.branch(AnalysisBranch)
# Branch runs with inherited history, returns its final_output
```

### Properties

#### `history: list[dict]`

Conversation history. Fully mutable in `on_step` - add, modify, or remove messages.

```python
def on_step(self, step):
    # Read history
    for msg in self.history:
        print(msg["role"], msg["content"][:50])

    # Mutate history - add a reminder
    self.history.append({
        "role": "user",
        "content": "Remember to cite sources."
    })

    # Mutate history - trim old messages
    if len(self.history) > 50:
        self.history = self.history[:1] + self.history[-40:]
```

---

## Step

Passed to `on_step` callback. Represents current loop state.

### Properties (Read)

| Property | Type | Description |
|----------|------|-------------|
| `counter` | `int` | Current step number (1-indexed) |
| `response` | `dict` | Raw model response |
| `tool_calls` | `list[ToolCall]` | Tool calls made this step |

### Properties (Read/Write - Mutable)

| Property | Type | Description |
|----------|------|-------------|
| `tool_results` | `list[ToolResult]` | Results from tool execution. Mutable - modify before next model call. |

### Properties (Read/Write)

| Property | Type | Description |
|----------|------|-------------|
| `model` | `str \| dict` | Model for next iteration |
| `temperature` | `float` | Temperature for next iteration |
| `max_tokens` | `int` | Max tokens for next iteration |
| `tools` | `list[Callable]` | Tools for next iteration |

### Methods

#### `add_tool(tool: Callable) -> None`

Add a tool for remaining steps.

```python
step.add_tool(new_search_function)
```

#### `remove_tool(name: str) -> None`

Remove a tool by name.

```python
step.remove_tool("search_web")
```

#### `finish(**kwargs) -> None`

Force completion with provided output. Arguments must match `final_output` schema.

```python
# If final_output = AnswerOutput(answer: str, confidence: float, sources: list[str])
step.finish(answer="Done", confidence=0.9, sources=[])
```

**Note**: To abort with an error, raise an exception instead:

```python
def on_step(self, step):
    if error_condition:
        raise ValueError("Cannot proceed without API key")
```

---

## ToolCall

Represents a tool call from the model.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Unique call identifier (e.g., `"call_abc123"`) |
| `name` | `str` | Tool name (e.g., `"search_web"`) |
| `arguments` | `dict` | Arguments passed to tool (e.g., `{"query": "python", "max_results": 5}`) |

### Example

```python
def on_step(self, step):
    for call in step.tool_calls:
        print(f"Tool: {call.name}")
        print(f"Args: {call.arguments}")
        # Tool: search_web
        # Args: {'query': 'python', 'max_results': 5}
```

---

## ToolResult

Represents the result of a tool execution.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `id` | `str` | Matches `ToolCall.id` (e.g., `"call_abc123"`) |
| `name` | `str` | Tool name (e.g., `"search_web"`) |
| `output` | `Any` | Return value from tool function |
| `error` | `str \| None` | Error message if tool raised an exception |

### Example

```python
def on_step(self, step):
    for result in step.tool_results:
        print(f"Tool: {result.name}")
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Output: {result.output}")
        # Tool: search_web
        # Output: ['result1', 'result2', 'result3']
```

---

## StreamChunk

Passed to `on_stream` callback.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `content` | `str \| None` | Text content chunk (chain-of-thought reasoning) |
| `partial` | `Partial[T] \| None` | Partially parsed `final_output` when streaming `__finish__` call |
| `tool_call` | `dict \| None` | Partial tool call data (for non-`__finish__` tools) |
| `done` | `bool` | True if final chunk |

### Partial[T] Type

`Partial[T]` is a generated type where all fields of `T` become `Optional`. This allows accessing fields as they stream in:

```python
from acorn import Partial
from pydantic import BaseModel

class MyOutput(BaseModel):
    name: str
    count: int
    summary: str

# Partial[MyOutput] is equivalent to:
class PartialMyOutput(BaseModel):
    name: str | None
    count: int | None
    summary: str | None
```

Usage in `on_stream`:

```python
def on_stream(self, chunk):
    if chunk.partial:
        # Fields appear as they're generated
        if chunk.partial.name:
            print(f"Name: {chunk.partial.name}")
        if chunk.partial.summary:
            print(f"Summary: {chunk.partial.summary}")
```

---

## @tool Decorator

Marks a function as a tool with automatic schema generation. Can be used on standalone functions or module methods.

### Standalone Function

```python
from acorn import tool

@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for information.

    Args:
        query: The search query
        max_results: Maximum results to return
    """
    return [...]
```

### Module Method

```python
class MyAgent(module):
    @tool
    def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web for information."""
        # Has access to self (module instance)
        return [...]
```

When used on methods, `self` is automatically excluded from the tool schema.

### Tool Resolution

Acorn collects tools from:
1. `tools = [...]` class attribute
2. Methods decorated with `@tool`

Both are concatenated. Name conflicts raise `ToolConflictError`.

### Schema Generation

| Source | Maps To |
|--------|---------|
| Function name | Tool name |
| Docstring first line | Tool description |
| Parameter type hints | Parameter types |
| Docstring Args section | Parameter descriptions |
| Default values | Optional parameters |
| Return type hint | (informational) |

### Supported Types

- `str`, `int`, `float`, `bool`
- `list[T]`, `dict[K, V]`
- `Optional[T]`, `T | None`
- Pydantic models
- Enums

---

## Exceptions

### AcornError

Base exception for all acorn errors.

### ParseError

Raised when output validation fails after all retries.

```python
try:
    result = agent(question="...")
except ParseError as e:
    print(f"Could not parse: {e.raw_output}")
```

### BranchError

Raised when a manual branch fails.

```python
try:
    result = self.branch(RiskyBranch)
except BranchError as e:
    print(f"Branch failed: {e}")
```

### ToolConflictError

Raised when a tool name conflict is detected (same name in `tools` attribute and `@tool` method).

```python
class BadAgent(module):
    tools = [search]  # Has a tool named "search"

    @tool
    def search(self, query: str) -> str:  # Conflict!
        ...

# Raises: ToolConflictError: Tool 'search' defined in both tools attribute and as method
```

---

## Branches

Branches are modules that extend their parent. See `branching.md` for full details.

### Branch Inheritance

When a branch is spawned:

| Aspect | Behavior |
|--------|----------|
| System prompt | Parent's prompt + branch's docstring |
| Tools | Parent's tools + branch's tools + `__finish__` |
| History | Full parent history inherited |
| Trigger | Tool call with no arguments |
| Output | Structured `final_output` (required) |

### Defining a Branch

```python
class AnalysisBranch(module):
    """
    Analyze the data discussed in the conversation.
    Focus on patterns and anomalies.
    """
    # Docstring = additional prompt appended to parent's

    tools = [statistical_analysis]  # Added to parent's tools
    final_output = AnalysisResult    # Required
    # No initial_input - context from inherited history
```

### Registering Branches

```python
class MainAgent(module):
    branches = {
        "analyze": AnalysisBranch,
        "verify": VerifyBranch,
    }
    # Generates tools: analyze(), verify()
```

### Custom Description

```python
branches = {
    "analyze": {
        "module": AnalysisBranch,
        "description": "Deep statistical analysis of discussed data",
    },
}
```

---

## Full Example

```python
from acorn import module, tool
from pydantic import BaseModel, Field
from pathlib import Path

# --- Tools ---

@tool
def search(query: str, limit: int = 5) -> list[str]:
    """Search for information online."""
    return ["result1", "result2"]

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)

# --- Branch ---

class FactCheckOutput(BaseModel):
    is_true: bool
    evidence: list[str]
    confidence: float

class FactChecker(module):
    """
    Verify claims discussed in the conversation.
    Search for evidence and assess truthfulness.
    """
    # Docstring = additional prompt appended to parent's system_prompt
    # No initial_input - context comes from inherited history

    temperature = 0.1
    max_steps = 5
    final_output = FactCheckOutput
    tools = [search]  # Added to parent's tools

# --- Main Agent ---

class QuestionInput(BaseModel):
    question: str = Field(description="Question to answer")

class AnswerOutput(BaseModel):
    answer: str
    confidence: float
    sources: list[str]

class ResearchAgent(module):
    """
    You are a research assistant. Answer questions thoroughly.
    Use fact_check when you need to verify important claims.
    """
    # Docstring used as system_prompt

    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.7
    max_tokens = 4096
    max_steps = 15
    parse_retries = 2

    initial_input = QuestionInput
    final_output = AnswerOutput

    tools = [search, calculate]
    branches = {"fact_check": FactChecker}
    # fact_check() tool auto-generated with no parameters
    # When called, FactChecker inherits:
    #   - System prompt: ResearchAgent's docstring + FactChecker's docstring
    #   - Tools: [search, calculate, search, __finish__] (merged)
    #   - History: Full parent history

    cache = [
        {"location": "message", "role": "system"},
    ]

    def on_step(self, step):
        print(f"[Step {step.counter}] Tools called: {[t.name for t in step.tool_calls]}")

        # Reduce temperature after step 10
        if step.counter > 10:
            step.temperature = 0.3

        # Mutate tool results before next model call
        for result in step.tool_results:
            if len(str(result.output)) > 5000:
                result.output = str(result.output)[:5000] + "..."

        return step

    def on_stream(self, chunk):
        if chunk.content:
            print(chunk.content, end="", flush=True)

# --- Usage ---

if __name__ == "__main__":
    agent = ResearchAgent()

    try:
        result = agent(question="What year was Python created and who created it?")
        print(f"\nAnswer: {result.answer}")
        print(f"Confidence: {result.confidence}")
        print(f"Sources: {result.sources}")
    except ParseError as e:
        print(f"Output parsing failed: {e}")
```
