# Core Concepts

This document defines the foundational building blocks of acorn.

## Module

The `module` is the first-class citizen of acorn. It encapsulates all configuration for an LLM agent: model settings, tools, input/output schemas, and lifecycle hooks.

### Basic Structure

```python
from acorn import module

class MyAgent(module):
    # Model configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.7
    max_tokens = 4096

    # Input/Output schemas
    initial_input = [...]
    final_output = [...]

    # Tools available to the agent
    tools = [...]

    # Optional: branches this module can spawn
    branches = {}

    # Optional: caching configuration
    cache = [...]

    # Lifecycle hooks
    def on_step(self, step):
        ...

    def on_stream(self, chunk):
        ...
```

### Invocation

Modules are callable. The `__call__` method accepts arguments matching the `initial_input` schema and returns a result matching the `final_output` schema.

```python
agent = MyAgent()
result = agent(question="What is the capital of France?")
# result is a Pydantic model matching final_output schema
```

---

## Model Configuration

### model

The model identifier string, passed directly to LiteLLM. Supports any provider LiteLLM supports.

```python
model = "anthropic/claude-sonnet-4-5-20250514"
model = "openai/gpt-4o"
model = "ollama/llama3"
```

Can also be a dict for advanced LiteLLM configuration:

```python
model = {
    "model": "anthropic/claude-sonnet-4-5-20250514",
    "api_key": "...",
    "base_url": "..."
}
```

### temperature

Float between 0 and 2. Controls randomness in model output.

```python
temperature = 0.7
```

### max_tokens

Maximum tokens in model response.

```python
max_tokens = 4096
```

### max_steps

Maximum iterations of the agentic loop. Default: `None` (single-turn mode).

- `None` - Single turn: input → `__finish__` call → output. Tools allowed but no loop.
- `N` - Up to N iterations. At step N, `__finish__` is forced via `tool_choice`.

```python
max_steps = None  # Default - single turn
max_steps = 10    # Agentic loop, up to 10 steps
```

### parse_retries

Number of times to retry when model output doesn't match expected schema. Default: 2.

```python
parse_retries = 3
```

---

## System Prompt

The system prompt provides instructions to the model. It should be written in Markdown format.

### Docstring (Recommended)

The simplest approach - use the class docstring as the system prompt:

```python
class MyAgent(module):
    """
    You are a helpful research assistant.

    ## Guidelines
    - Be concise
    - Cite sources
    """
    # The docstring above becomes the system_prompt
```

### String Attribute

Explicit `system_prompt` attribute takes precedence over docstring:

```python
class MyAgent(module):
    """This docstring is ignored when system_prompt is set."""

    system_prompt = """
    You are a helpful research assistant.

    ## Guidelines
    - Be concise
    - Cite sources
    """
```

### File Path

```python
from pathlib import Path

class MyAgent(module):
    system_prompt = Path("prompts/researcher.md")
```

### Dynamic (Method)

```python
from datetime import date

class MyAgent(module):
    def system_prompt(self):
        return f"""
        You are a helpful assistant.
        Today's date is {date.today()}.
        """
```

---

## Input/Output Schemas

Schemas are defined using Pydantic models or Python typing. The user never writes XML - acorn handles serialization internally when communicating with the model.

### initial_input

Defines the arguments accepted when calling the module.

```python
from pydantic import BaseModel, Field

class QuestionInput(BaseModel):
    question: str = Field(description="The question to answer")
    context: str | None = Field(default=None, description="Optional context")

class MyAgent(module):
    initial_input = QuestionInput
```

### final_output

Defines the structure returned when the module completes (via `__finish__` tool).

```python
from pydantic import BaseModel, Field

class AnswerOutput(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="Sources used")

class MyAgent(module):
    final_output = AnswerOutput
```

When the model calls `__finish__`, the tool call arguments (JSON) are validated against this schema via Pydantic. The module returns an instance of this Pydantic model. This is the **only** way to produce structured output - there is no XML output mode for normal operation.

---

## Tools

Tools are defined as decorated functions with type hints. Acorn auto-generates the tool schema from the function signature and docstring.

### Defining Tools

```python
from acorn import tool

@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """
    Search the web for information.

    Args:
        query: The search query
        max_results: Maximum number of results to return

    Returns:
        List of search results with title, url, and snippet
    """
    # implementation
    return results
```

### Registering Tools

**Option 1: Class attribute**

```python
class MyAgent(module):
    tools = [search_web, read_file, calculate]
```

**Option 2: Methods with @tool decorator**

```python
class MyAgent(module):
    @tool
    def search_web(self, query: str, max_results: int = 5) -> list[dict]:
        """Search the web for information."""
        # implementation - has access to self
        return results

    @tool
    def calculate(self, expression: str) -> float:
        """Evaluate a math expression."""
        return eval(expression)
```

**Option 3: Both (concatenated)**

```python
class MyAgent(module):
    tools = [external_api_tool]  # External tools

    @tool
    def internal_tool(self, data: str) -> str:
        """Tool with access to module state."""
        return self.process(data)
```

When both are present, acorn concatenates them. If there's a name conflict (same tool name in both), an error is raised.

### The __finish__ Tool

Every module automatically has a `__finish__` tool. When the model calls this tool with data matching `final_output` schema, the loop terminates and the module returns the result.

The model sees this as a regular tool:

```
__finish__(answer="Paris", confidence=0.95, sources=["wikipedia.org"])
```

### Tool Schema Generation

Acorn generates tool schemas from:
- Function name → tool name
- Docstring first line → tool description
- Parameter type hints → parameter types
- Parameter docstrings (Args section) → parameter descriptions
- Default values → optional parameters

---

## Caching

Provider-level prompt caching (Anthropic-style cache breakpoints).

```python
class MyAgent(module):
    cache = [
        {
            "location": "message",
            "role": "system",
        },
        {
            "location": "message",
            "index": 0,
        }
    ]
```

This tells the provider to cache:
1. The system message
2. The first user message

Caching reduces latency and cost for repeated prefixes.

---

## History

The conversation history is fully accessible and mutable within the module.

```python
def on_step(self, step):
    print(self.history)  # List of messages
```

History is a list of message dicts that you can read and modify:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
    ...
]
```

### Mutating History

You can directly modify `self.history` in `on_step` to control what the model sees:

```python
def on_step(self, step):
    # Append a reminder message
    self.history.append({
        "role": "user",
        "content": "Remember to be concise in your response."
    })

    # Modify an existing message
    for msg in self.history:
        if msg["role"] == "system":
            msg["content"] += "\n\nAdditional context: ..."

    # Remove old messages to manage context length
    if len(self.history) > 50:
        # Keep system message + last 40 messages
        self.history = self.history[:1] + self.history[-40:]

    return step
```

### Mutating Tool Results

The `step.tool_results` list is also mutable. Modify it before returning from `on_step` to change what the model sees as tool output:

```python
def on_step(self, step):
    for result in step.tool_results:
        if result.name == "search_web":
            # Truncate large outputs
            if len(str(result.output)) > 5000:
                result.output = result.output[:5000] + "... (truncated)"

    return step
```
