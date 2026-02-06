# Module

The `Module` class is the foundation of acorn. It encapsulates everything needed to build an LLM agent: model configuration, input/output schemas, tools, and execution logic.

## What is Module?

A Module is a reusable component that wraps an LLM with structured inputs and outputs. Think of it as a function that:
- Takes typed arguments (validated with Pydantic)
- Calls an LLM with those arguments
- Returns typed results (also validated with Pydantic)

Unlike raw LLM calls that work with strings, Modules give you type safety and validation at both ends.

## Single-Turn vs Multi-Turn

Modules can run in two modes:

**Single-turn** (default): One LLM call that immediately produces output. The LLM must call the `__finish__` tool with your output schema.

```python
class Summarizer(Module):
    # max_steps not set = single-turn
    initial_input = Input
    final_output = Output
```

**Multi-turn** (agentic loop): The LLM can call tools multiple times across multiple steps before finishing.

```python
class ResearchAgent(Module):
    max_steps = 5  # Up to 5 iterations
    initial_input = Input
    final_output = Output
    tools = [search, analyze]
```

When `max_steps` is set, the agent runs in a loop:
1. LLM decides which tool to call
2. Tool executes and returns result
3. Result goes back to LLM
4. Repeat until LLM calls `__finish__` or max steps reached

## Basic Usage

Define a Module by subclassing and setting class attributes:

```python
from pydantic import BaseModel, Field
from acorn import Module

class Input(BaseModel):
    text: str = Field(description="Text to analyze")

class Output(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")

class SentimentAnalyzer(Module):
    """Analyze sentiment of text."""
    
    initial_input = Input
    final_output = Output
    temperature = 0.3

# Use it
analyzer = SentimentAnalyzer()
result = analyzer(text="I love this product!")
print(result.sentiment)  # "positive"
print(result.confidence)  # 0.95
```

## Model Configuration

Control which LLM to use and how it behaves:

### model

The LLM to use. Defaults to Claude Sonnet 4.5.

```python
class MyModule(Module):
    model = "anthropic/claude-sonnet-4-5-20250514"  # Default
    model = "openai/gpt-4o"
    model = "ollama/llama3"
```

For advanced configuration, use a dict:

```python
model = {
    "id": "anthropic/claude-sonnet-4-5-20250514",
    "vertex_location": "us-central1",
    "reasoning": "high"  # Extended thinking mode
}
```

### temperature

Controls randomness in responses (0.0-1.0). Lower = more deterministic.

```python
temperature = 0.3  # Precise, consistent outputs
temperature = 0.7  # Default, balanced
temperature = 1.0  # Creative, varied outputs
```

### max_tokens

Maximum tokens the LLM can generate in a single response.

```python
max_tokens = 4096  # Default
max_tokens = 1000  # Shorter responses
```

### max_steps

Maximum iterations in the agentic loop. `None` means single-turn.

```python
max_steps = None  # Single-turn (default)
max_steps = 5     # Up to 5 iterations
max_steps = 20    # Longer tasks
```

### cache

Enable provider-level prompt caching to reduce latency and cost:

```python
class MyModule(Module):
    cache = True  # Cache system + first user message
```

Or specify custom cache points:

```python
cache = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": 0}
]
```

## Schemas

Define what goes in and what comes out using Pydantic models.

### initial_input

Validates arguments passed when calling the module.

```python
class Input(BaseModel):
    question: str = Field(description="Question to answer")
    context: str | None = Field(default=None, description="Optional context")

class MyModule(Module):
    initial_input = Input

# Call with validated arguments
module = MyModule()
result = module(question="What is Python?", context="Programming")
```

If `initial_input` is not set, the module accepts any keyword arguments.

### final_output

Defines the structure of the result. Required for single-turn, optional for multi-turn.

```python
class Output(BaseModel):
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score")
    sources: list[str] = Field(description="Sources used")

class MyModule(Module):
    final_output = Output
```

When the LLM calls `__finish__`, the arguments are validated against this schema. If validation fails, acorn automatically retries with an error message.

For multi-turn modules without structured output, set `final_output = None`:

```python
class TaskExecutor(Module):
    max_steps = 10
    final_output = None  # No structured output
    tools = [do_task_a, do_task_b]

executor = TaskExecutor()
result = executor()  # Returns None after executing tools
```

## System Prompt

Instructions for the LLM. Four ways to set it:

**1. Docstring (recommended)**

```python
class MyModule(Module):
    """You are a helpful assistant.
    
    Be concise and cite sources.
    """
```

**2. String attribute**

```python
class MyModule(Module):
    system_prompt = "You are a helpful assistant."
```

**3. File path**

```python
from acorn import Path

class MyModule(Module):
    system_prompt = Path("prompts/assistant.md")
```

**4. Method (dynamic)**

```python
from datetime import date

class MyModule(Module):
    def system_prompt(self):
        return f"You are a helpful assistant. Today is {date.today()}."
```

## Tools

Functions the LLM can call to gather information or take actions.

### Defining Tools

Use the `@tool` decorator on functions or methods:

```python
from acorn import tool

@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum number of results
    """
    # Your implementation
    return results
```

Acorn generates the tool schema from:
- Function name → tool name
- First line of docstring → tool description
- Type hints → parameter types
- Args section → parameter descriptions
- Default values → optional parameters

### Adding Tools to Module

**Option 1: Class attribute**

```python
class MyModule(Module):
    tools = [search_web, calculate, read_file]
```

**Option 2: Decorated methods**

```python
class MyModule(Module):
    @tool
    def search_web(self, query: str) -> list:
        """Search the web."""
        # Has access to self
        return self._search_api(query)
    
    @tool
    def analyze(self, data: str) -> str:
        """Analyze data."""
        return self._analyze(data)
```

**Option 3: Both**

```python
class MyModule(Module):
    tools = [external_tool]  # External functions
    
    @tool
    def internal_tool(self, data: str) -> str:
        """Tool with access to module state."""
        return self.process(data)
```

Both lists are combined. If there's a name conflict, acorn raises `ToolConflictError`.

### The __finish__ Tool

Every module automatically gets a `__finish__` tool. When the LLM calls it, the module terminates and returns the result.

The tool's parameters match your `final_output` schema:

```python
class Output(BaseModel):
    answer: str
    confidence: float

# LLM sees this tool:
# __finish__(answer="Paris", confidence=0.95)
```

If `final_output = None`, `__finish__` takes no parameters:

```python
# LLM sees:
# __finish__()
```

## Lifecycle Hooks

Customize behavior at key points in execution.

### on_step

Called after each step in the agentic loop. Use it to:
- Track progress
- Modify tool results
- Add/remove tools dynamically
- Terminate early
- Adjust parameters

```python
class MyModule(Module):
    max_steps = 10
    
    def on_step(self, step):
        print(f"Step {step.counter}")
        print(f"Tools called: {[tc.name for tc in step.tool_calls]}")
        
        # Modify tool results before next LLM call
        for result in step.tool_results:
            if len(str(result.output)) > 5000:
                result.output = str(result.output)[:5000] + "..."
        
        # Dynamic tool management
        if step.counter > 5:
            step.add_tool(advanced_tool)
        
        # Early termination
        if self._is_done():
            step.finish(answer="Done early", confidence=1.0)
        
        return step
```

The `step` object has:
- `counter`: Current step number (1-indexed)
- `tool_calls`: List of `ToolCall` objects
- `tool_results`: List of `ToolResult` objects (mutable)
- `model`, `temperature`, `max_tokens`: Current settings (can modify)
- `tools`: Current tool list (can modify)

### on_stream

Called when streaming is enabled. Receives chunks as they arrive:

```python
class MyModule(Module):
    stream = True
    
    def on_stream(self, chunk):
        # Text content (chain-of-thought)
        if chunk.content:
            print(chunk.content, end="", flush=True)
        
        # Partial structured output
        if chunk.partial:
            if chunk.partial.answer:
                print(f"\nAnswer: {chunk.partial.answer}")
```

## Advanced Configuration

### metadata

Arbitrary metadata passed to LiteLLM for tracking:

```python
class MyModule(Module):
    metadata = {
        "user_id": "user123",
        "session_id": "session456"
    }
```

### xml_input_root / xml_output_root

Customize XML element names for input/output serialization:

```python
class MyModule(Module):
    xml_input_root = "query"     # Default: "input"
    xml_output_root = "response"  # Default: "output"
```

### max_parse_retries

Number of times to retry when output validation fails:

```python
class MyModule(Module):
    max_parse_retries = 3  # Default: 2
```

## Common Patterns

### Research Assistant

Multi-turn agent that gathers information before answering:

```python
class ResearchAgent(Module):
    """Research assistant that uses tools to answer questions."""
    
    class Input(BaseModel):
        question: str
    
    class Output(BaseModel):
        answer: str
        sources: list[str]
    
    initial_input = Input
    final_output = Output
    max_steps = 5
    temperature = 0.3
    
    @tool
    def search(self, query: str) -> list:
        """Search for information."""
        return search_api(query)
    
    @tool
    def analyze(self, data: str) -> str:
        """Analyze collected data."""
        return analyze_data(data)
    
    def on_step(self, step):
        print(f"Step {step.counter}: {[tc.name for tc in step.tool_calls]}")
        return step
```

### Data Processor

Single-turn module for data transformation:

```python
class DataProcessor(Module):
    """Transform data from one format to another."""
    
    class Input(BaseModel):
        data: dict
        target_format: str
    
    class Output(BaseModel):
        transformed_data: dict
        validation_errors: list[str]
    
    initial_input = Input
    final_output = Output
    temperature = 0.1  # Deterministic
```

### Task Executor

Multi-turn without structured output:

```python
class TaskExecutor(Module):
    """Execute a series of tasks using tools."""
    
    max_steps = 10
    final_output = None  # No structured output
    
    @tool
    def create_file(self, path: str, content: str) -> str:
        """Create a file."""
        Path(path).write_text(content)
        return f"Created {path}"
    
    @tool
    def send_email(self, to: str, subject: str) -> str:
        """Send an email."""
        send_mail(to, subject)
        return f"Sent email to {to}"

executor = TaskExecutor()
result = executor()  # Returns None after executing tools
```

## History

The conversation history is accessible and mutable via `self.history`:

```python
def on_step(self, step):
    # Read history
    print(f"History has {len(self.history)} messages")
    
    # Add a message
    self.history.append({
        "role": "user",
        "content": "Remember to be concise."
    })
    
    # Trim old messages to manage context
    if len(self.history) > 50:
        self.history = self.history[:1] + self.history[-40:]
    
    return step
```

History is a list of message dicts:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."}
]
```

## Error Handling

### ParseError

Raised when output validation fails after all retries:

```python
from acorn import ParseError

try:
    result = module(text="...")
except ParseError as e:
    print(f"Failed to parse output: {e}")
    print(f"Raw output: {e.raw_output}")
```

### ToolConflictError

Raised when duplicate tool names are detected:

```python
from acorn import ToolConflictError

class BadModule(Module):
    tools = [search]  # External tool named "search"
    
    @tool
    def search(self, query: str) -> str:  # Conflict!
        pass

# Raises: ToolConflictError: Duplicate tool name: search
```

### AcornError

Base exception for all acorn errors:

```python
from acorn import AcornError

try:
    result = module(text="...")
except AcornError as e:
    print(f"Acorn error: {e}")
```

## Best Practices

**Start with single-turn**: Use `max_steps = None` unless you need tool calling across multiple steps.

**Keep schemas simple**: Complex nested schemas are harder for LLMs to fill correctly. Flatten when possible.

**Write clear tool descriptions**: The first line of the docstring is what the LLM sees. Make it count.

**Use field descriptions**: Pydantic field descriptions guide the LLM on what to provide.

**Set appropriate temperature**: Lower (0.1-0.3) for deterministic tasks, higher (0.7-1.0) for creative tasks.

**Monitor with on_step**: Track tool usage and results to understand agent behavior.

**Validate early**: Use Pydantic validators on input/output schemas to catch issues.

**Handle large outputs**: Truncate tool results in `on_step` to avoid context limits.

## Next Steps

- See [Getting Started](getting-started.md) for complete examples
- Check the [API Reference](../specs/api-reference.md) for detailed attribute documentation
- Explore [examples/](../examples/) for real implementations