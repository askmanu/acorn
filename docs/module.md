# Module

The `Module` class is the core building block of acorn. It encapsulates everything needed to create an LLM agent: model configuration, input/output schemas, tools, and execution behavior.

## What is Module?

A Module defines a reusable LLM agent with:
- **Model settings**: Which LLM to use and how to configure it
- **Structured I/O**: Pydantic schemas for inputs and outputs
- **Tools**: Functions the agent can call to gather information or take actions
- **Execution mode**: Single-turn (one call) or multi-turn (agentic loop)
- **Lifecycle hooks**: Callbacks to observe or modify behavior

Think of Module as a template for an agent. You define the class once, then create instances and call them with different inputs.

## Basic Usage

```python
from pydantic import BaseModel, Field
from acorn import Module

class Input(BaseModel):
    text: str = Field(description="Text to analyze")

class Output(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="0 to 1")

class SentimentAnalyzer(Module):
    """Analyze the sentiment of text."""
    
    initial_input = Input
    final_output = Output
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.3

# Use it
analyzer = SentimentAnalyzer()
result = analyzer(text="I love this product!")
print(result.sentiment)  # "positive"
print(result.confidence)  # 0.95
```

## Single-Turn vs Multi-Turn

Modules operate in two modes:

### Single-Turn Mode

Default behavior (`max_steps = None`). The agent makes one LLM call and returns structured output.

```python
class Classifier(Module):
    """Classify input text."""
    initial_input = Input
    final_output = Output
    # max_steps not set = single-turn
```

Flow:
1. Validate input against `initial_input` schema
2. Call LLM with input and available tools
3. LLM calls `__finish__` tool with structured output
4. Validate output against `final_output` schema
5. Return validated Pydantic model

### Multi-Turn Mode

Enable agentic behavior by setting `max_steps` to a positive integer. The agent can iterate, calling tools and reasoning across multiple steps.

```python
class Researcher(Module):
    """Research a topic using tools."""
    initial_input = Input
    final_output = Output
    max_steps = 10
    tools = [search_web, read_file]
```

Flow:
1. Validate input
2. Start loop (up to `max_steps` iterations)
3. Call LLM with current history
4. Execute any tool calls
5. Add results to history
6. Call `on_step` hook if defined
7. If LLM calls `__finish__`, validate and return output
8. Otherwise, continue loop
9. If max steps reached, force termination

## Model Configuration

### model

The LLM to use. Can be a string or dict.

**String format** (most common):

```python
model = "anthropic/claude-sonnet-4-5-20250514"
model = "openai/gpt-4o"
model = "vertex_ai/gemini-1.5-pro"
```

**Dict format** (advanced):

```python
model = {
    "id": "anthropic/claude-sonnet-4-5-20250514",
    "vertex_location": "us-central1",
    "vertex_credentials": "path/to/creds.json",
    "reasoning": True  # Enable extended thinking
}
```

The dict format supports:
- `id` (required): Model identifier
- `vertex_location`: Google Cloud region for Vertex AI
- `vertex_credentials`: Path to credentials file
- `reasoning`: Enable extended thinking mode (`True`, `"low"`, `"medium"`, or `"high"`)

### temperature

Controls randomness in responses. Range: 0.0 to 1.0.

```python
temperature = 0.7  # Default - balanced
temperature = 0.0  # Deterministic
temperature = 1.0  # More creative
```

Lower values make output more focused and deterministic. Higher values increase randomness and creativity.

### max_tokens

Maximum tokens the model can generate in a single response.

```python
max_tokens = 4096  # Default
max_tokens = 8192  # Longer responses
```

### max_steps

Maximum iterations in the agentic loop. Controls execution mode.

```python
max_steps = None  # Default - single-turn
max_steps = 5     # Up to 5 iterations
max_steps = 20    # More complex tasks
```

If the agent reaches `max_steps` without calling `__finish__`, acorn forces termination and requires a final response.

## Prompts and Schemas

### system_prompt

Instructions for the LLM. Can be a string, file path, method, or docstring.

**Docstring** (recommended):

```python
class Agent(Module):
    """You are a helpful assistant.
    
    Follow these guidelines:
    - Be concise
    - Cite sources
    - Ask clarifying questions when needed
    """
    # Docstring becomes the system prompt
```

**String attribute**:

```python
system_prompt = """
You are a research assistant.
Focus on accuracy and cite sources.
"""
```

**File path**:

```python
from pathlib import Path

system_prompt = Path("prompts/assistant.md")
```

**Method** (dynamic):

```python
from datetime import date

def system_prompt(self):
    return f"""
    You are an assistant.
    Today's date: {date.today()}
    """
```

### initial_input

Pydantic model defining the input schema. Arguments passed to `__call__` must match this schema.

```python
class Input(BaseModel):
    question: str = Field(description="Question to answer")
    context: str | None = Field(default=None, description="Optional context")

class Agent(Module):
    initial_input = Input

# Call with matching arguments
agent = Agent()
result = agent(question="What is Python?", context="Programming languages")
```

If `initial_input` is `None`, the module accepts any keyword arguments.

### final_output

Pydantic model defining the output schema. The LLM must call `__finish__` with data matching this schema.

```python
class Output(BaseModel):
    answer: str = Field(description="The answer")
    confidence: float = Field(description="0 to 1")
    sources: list[str] = Field(description="Sources consulted")

class Agent(Module):
    final_output = Output
```

For single-turn modules, `final_output` is required. For multi-turn modules, it's optional - set to `None` if you don't need structured output.

```python
class ToolExecutor(Module):
    """Execute tools without returning structured output."""
    max_steps = 5
    final_output = None  # No structured output needed
    tools = [log_action, save_data]
```

## Tools

Tools are functions the LLM can call. Define them with the `@tool` decorator or as methods.

### Standalone Tools

```python
from acorn import tool

@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.
    
    Args:
        query: Search query
        max_results: Maximum results to return
    """
    # Implementation
    return results

class Agent(Module):
    tools = [search_web]
```

### Method Tools

Tools defined as methods have access to `self`:

```python
class Agent(Module):
    @tool
    def search(self, query: str) -> list[str]:
        """Search using the agent's configuration."""
        # Access module state via self
        return self._search_engine.search(query)
```

### Tool Schema Generation

Acorn generates tool schemas from:
- Function name → tool name
- Docstring first line → description
- Type hints → parameter types
- Docstring Args section → parameter descriptions
- Default values → optional parameters

### The __finish__ Tool

Every module automatically has a `__finish__` tool. When the LLM calls it, the module validates the arguments against `final_output` and returns the result.

The LLM sees `__finish__` as a regular tool with parameters matching your output schema.

## Advanced Configuration

### metadata

LiteLLM metadata for tracking and logging.

```python
metadata = {
    "user_id": "user_123",
    "session_id": "session_456",
    "tags": ["production", "research"]
}
```

### cache

Provider-level prompt caching to reduce latency and costs.

```python
# Enable default caching (system message + first user message)
cache = True

# Disable caching
cache = False

# Custom cache breakpoints
cache = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": 0}
]
```

Each cache entry specifies where to insert a cache breakpoint:
- `location`: Must be `"message"`
- `role`: Message role to cache (`"system"`, `"user"`, etc.)
- `index`: Message index to cache (0-based)

### XML Configuration

Control XML serialization for input/output (used internally when communicating with the LLM).

```python
xml_input_root = "input"   # Root element for input
xml_output_root = "output" # Root element for output
```

You typically don't need to change these unless you have specific prompt engineering requirements.

### max_parse_retries

Number of times to retry when output validation fails.

```python
max_parse_retries = 2  # Default
max_parse_retries = 5  # More retries for complex schemas
```

When the LLM's output doesn't match the schema, acorn sends an error message and asks it to try again.

### stream

Enable streaming responses. Requires defining an `on_stream` callback.

```python
class Agent(Module):
    stream = True
    
    def on_stream(self, chunk):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.partial:
            # Access partial structured output as it streams
            print(f"Partial: {chunk.partial}")
```

## Lifecycle Hooks

### on_step

Called after each step in multi-turn mode. Use it to observe or modify behavior.

```python
def on_step(self, step):
    """Called after each agentic loop iteration."""
    print(f"Step {step.counter}")
    print(f"Tools called: {[tc.name for tc in step.tool_calls]}")
    
    # Modify configuration for next step
    if step.counter > 5:
        step.temperature = 0.3  # Reduce randomness
    
    # Add or remove tools
    if some_condition:
        step.add_tool(new_tool)
        step.remove_tool("old_tool")
    
    # Force completion
    if error_condition:
        step.finish(answer="Error occurred", confidence=0.0, sources=[])
    
    return step
```

The `step` object provides:
- `counter`: Step number (1-indexed)
- `tool_calls`: List of `ToolCall` objects
- `tool_results`: List of `ToolResult` objects (mutable)
- `model`, `temperature`, `max_tokens`: Configuration (mutable)
- `add_tool(tool)`: Add a tool for remaining steps
- `remove_tool(name)`: Remove a tool by name
- `finish(**kwargs)`: Force completion with output

### on_stream

Called for each chunk when streaming is enabled.

```python
def on_stream(self, chunk):
    """Handle streaming responses."""
    # Text content (chain-of-thought)
    if chunk.content:
        print(chunk.content, end="", flush=True)
    
    # Partial structured output (as __finish__ streams)
    if chunk.partial:
        if chunk.partial.answer:
            print(f"\nAnswer: {chunk.partial.answer}")
    
    # Check if done
    if chunk.done:
        print("\nStreaming complete")
```

## History

The conversation history is accessible via `self.history`. It's a list of message dictionaries that you can read and modify.

```python
def on_step(self, step):
    # Read history
    for msg in self.history:
        print(f"{msg['role']}: {msg['content'][:50]}")
    
    # Add a message
    self.history.append({
        "role": "user",
        "content": "Remember to cite sources."
    })
    
    # Trim old messages
    if len(self.history) > 50:
        self.history = self.history[:1] + self.history[-40:]
    
    return step
```

History format:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
]
```

## Examples

### Example 1: Simple Classifier

```python
from pydantic import BaseModel, Field
from acorn import Module

class Input(BaseModel):
    text: str

class Output(BaseModel):
    category: str
    confidence: float

class Classifier(Module):
    """Classify text into categories."""
    initial_input = Input
    final_output = Output
    temperature = 0.2

classifier = Classifier()
result = classifier(text="This movie was amazing!")
```

### Example 2: Research Agent with Tools

```python
from acorn import Module, tool

@tool
def search_web(query: str) -> list[str]:
    """Search the web."""
    return ["result1", "result2"]

@tool
def read_file(path: str) -> str:
    """Read a file."""
    return open(path).read()

class Researcher(Module):
    """Research topics using available tools."""
    max_steps = 10
    tools = [search_web, read_file]
    
    class Input(BaseModel):
        topic: str
    
    class Output(BaseModel):
        summary: str
        sources: list[str]
    
    initial_input = Input
    final_output = Output
    
    def on_step(self, step):
        print(f"Step {step.counter}: {[tc.name for tc in step.tool_calls]}")
        return step

researcher = Researcher()
result = researcher(topic="Python history")
```

### Example 3: Streaming Responses

```python
class StreamingAgent(Module):
    """Agent with streaming enabled."""
    stream = True
    
    def on_stream(self, chunk):
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.done:
            print("\n[Done]")

agent = StreamingAgent()
result = agent(question="Explain Python")
```

## Common Patterns

### Reducing Temperature Over Time

```python
def on_step(self, step):
    if step.counter > 5:
        step.temperature = 0.3  # More focused after initial exploration
    return step
```

### Conditional Tool Availability

```python
def on_step(self, step):
    # Remove expensive tools after step 3
    if step.counter > 3:
        step.remove_tool("expensive_api_call")
    return step
```

### Managing Context Length

```python
def on_step(self, step):
    # Keep system message + last 30 messages
    if len(self.history) > 31:
        self.history = self.history[:1] + self.history[-30:]
    return step
```

### Truncating Large Tool Results

```python
def on_step(self, step):
    for result in step.tool_results:
        output_str = str(result.output)
        if len(output_str) > 5000:
            result.output = output_str[:5000] + "... (truncated)"
    return step
```

### Early Termination

```python
def on_step(self, step):
    if error_detected:
        step.finish(
            answer="Unable to complete task",
            confidence=0.0,
            sources=[]
        )
    return step
```

## Best Practices

**Use descriptive field descriptions**: The LLM sees these descriptions and uses them to understand what to provide.

```python
class Output(BaseModel):
    answer: str = Field(description="Detailed answer to the question")
    confidence: float = Field(description="Confidence score between 0 and 1")
```

**Set appropriate max_steps**: Too low and the agent can't complete complex tasks. Too high and costs increase.

**Use temperature wisely**: Lower for factual tasks (0.0-0.3), higher for creative tasks (0.7-1.0).

**Monitor with on_step**: Track what the agent is doing, especially during development.

**Validate tool outputs**: Use `on_step` to truncate or clean tool results before the next LLM call.

**Cache system prompts**: Use `cache = True` or custom cache breakpoints to reduce costs for repeated calls.

**Handle errors gracefully**: Use `max_parse_retries` and consider error handling in `on_step`.
