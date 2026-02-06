# Module

The `Module` class is the core building block of acorn. It encapsulates everything needed to build an LLM agent: model configuration, input/output schemas, tools, and execution behavior.

## What is Module?

A Module is a Python class that defines an LLM agent with structured inputs and outputs. When you create a Module, you specify:

- **Model settings** - Which LLM to use and how it should behave
- **Input schema** - What data your agent accepts
- **Output schema** - What data your agent returns
- **Tools** - Functions the LLM can call
- **System prompt** - Instructions for the LLM

Modules are callable. Pass your inputs as keyword arguments and get back a validated Pydantic model:

```python
from pydantic import BaseModel
from acorn import Module

class Summarizer(Module):
    """Summarize text concisely."""
    
    class Input(BaseModel):
        text: str
    
    class Output(BaseModel):
        summary: str
    
    initial_input = Input
    final_output = Output

summarizer = Summarizer()
result = summarizer(text="Long article...")
print(result.summary)  # Typed Pydantic model
```

## Single-Turn vs Multi-Turn

Modules operate in two modes:

### Single-Turn Mode

Default behavior when `max_steps` is not set (or `max_steps = None`). The LLM makes one call and returns structured output immediately.

```python
class Classifier(Module):
    """Classify sentiment as positive, negative, or neutral."""
    
    class Input(BaseModel):
        text: str
    
    class Output(BaseModel):
        sentiment: str
        confidence: float
    
    initial_input = Input
    final_output = Output
    # max_steps = None (default)
```

**When to use:** Tasks that require a single decision or transformation without external data gathering.

### Multi-Turn Mode

Enabled by setting `max_steps` to a number. The LLM can call tools multiple times in a loop, gathering information before producing the final output.

```python
class ResearchAgent(Module):
    """Research a topic using multiple tools."""
    
    max_steps = 5  # Up to 5 iterations
    
    @tool
    def search(self, query: str) -> list:
        """Search for information."""
        return search_api(query)
    
    @tool
    def analyze(self, data: str) -> str:
        """Analyze collected data."""
        return analysis_result
```

**When to use:** Tasks that require gathering information, iterative reasoning, or multiple actions.

## Configuration Attributes

### Model Configuration

#### model

The LLM to use. Accepts a string identifier or configuration dictionary.

**String format:**

```python
model = "anthropic/claude-sonnet-4-5-20250514"  # Default
model = "openai/gpt-4o"
model = "ollama/llama3"
```

**Dictionary format** for advanced configuration:

```python
model = {
    "id": "anthropic/claude-sonnet-4-5-20250514",
    "vertex_location": "us-central1",  # Optional: Vertex AI location
    "vertex_credentials": "path/to/creds.json",  # Optional: Vertex credentials
    "reasoning": "high"  # Optional: True, "low", "medium", or "high"
}
```

The model identifier is passed to LiteLLM, which supports any LLM provider.

#### temperature

Controls randomness in model outputs. Range: 0.0 to 1.0.

```python
temperature = 0.7  # Default
temperature = 0.0  # Deterministic
temperature = 1.0  # More creative
```

**Lower values** (0.0-0.3) produce consistent, focused outputs. Use for classification, structured extraction, or deterministic tasks.

**Higher values** (0.7-1.0) produce varied, creative outputs. Use for content generation or brainstorming.

#### max_tokens

Maximum tokens the model can generate in a single response.

```python
max_tokens = 4096  # Default
max_tokens = 8192  # Longer responses
```

Set this based on your expected output length. The model stops generating when it reaches this limit.

#### max_steps

Maximum iterations in the agentic loop. Controls execution mode.

```python
max_steps = None  # Default: single-turn mode
max_steps = 5     # Multi-turn: up to 5 iterations
max_steps = 10    # Multi-turn: up to 10 iterations
```

In multi-turn mode, the loop terminates when:
- The LLM calls `__finish__` with final output
- The step limit is reached (forces termination)

### Prompt and Schema

#### system_prompt

Instructions for the LLM. Can be a string, file path, or method.

**Using class docstring** (recommended):

```python
class MyAgent(Module):
    """You are a helpful research assistant.
    
    Follow these guidelines:
    - Be concise and accurate
    - Cite sources when available
    - Ask clarifying questions if needed
    """
```

**Using string attribute:**

```python
class MyAgent(Module):
    system_prompt = """You are a helpful research assistant.
    
    Follow these guidelines:
    - Be concise and accurate
    - Cite sources when available
    """
```

**Using file path:**

```python
from pathlib import Path

class MyAgent(Module):
    system_prompt = Path("prompts/assistant.md")
```

**Using method** for dynamic prompts:

```python
from datetime import date

class MyAgent(Module):
    def system_prompt(self):
        return f"""You are a helpful assistant.
        Today's date: {date.today()}
        """
```

#### initial_input

Pydantic model defining the input schema. Arguments passed when calling the module must match this schema.

```python
from pydantic import BaseModel, Field

class Input(BaseModel):
    question: str = Field(description="The question to answer")
    context: str | None = Field(default=None, description="Optional context")

class MyAgent(Module):
    initial_input = Input

# Usage
agent = MyAgent()
result = agent(question="What is Python?", context="Programming")
```

Set to `None` if your module accepts no structured input:

```python
class MyAgent(Module):
    initial_input = None  # No schema
```

#### final_output

Pydantic model defining the output schema. The module returns an instance of this model.

```python
class Output(BaseModel):
    answer: str = Field(description="The answer")
    confidence: float = Field(description="Confidence score 0-1")
    sources: list[str] = Field(description="Sources used")

class MyAgent(Module):
    final_output = Output

# Returns typed Pydantic model
result = agent(question="...")
print(result.answer)  # Type-safe access
print(result.confidence)
```

**Required for single-turn mode.** In multi-turn mode, you can set `final_output = None` if you don't need structured output:

```python
class ToolExecutor(Module):
    max_steps = 5
    final_output = None  # No structured output
    
    @tool
    def execute_action(self, action: str) -> str:
        """Execute an action."""
        return "done"

executor = ToolExecutor()
result = executor()  # Returns None after executing tools
```

### Tools

#### tools

List of tool functions available to the LLM. Tools can be defined externally or as methods.

**External tools:**

```python
from acorn import tool

@tool
def search_web(query: str, limit: int = 10) -> list:
    """Search the web for information.
    
    Args:
        query: The search query
        limit: Maximum number of results
    """
    return search_api(query, limit)

class MyAgent(Module):
    tools = [search_web]
```

**Method tools** with `@tool` decorator:

```python
class MyAgent(Module):
    @tool
    def search(self, query: str) -> list:
        """Search for information.
        
        Args:
            query: The search query
        """
        # Has access to self
        return self._search_internal(query)
```

**Combining both:**

```python
class MyAgent(Module):
    tools = [external_tool]  # External tools
    
    @tool
    def internal_tool(self, data: str) -> str:
        """Internal tool with access to module state."""
        return self.process(data)
```

Tool schemas are automatically generated from function signatures and docstrings. The LLM sees:
- Function name as tool name
- First line of docstring as description
- Type hints as parameter types
- Args section as parameter descriptions

#### @tool decorator

Marks a function as a tool with automatic schema generation.

```python
from acorn import tool

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression.
    
    Args:
        expression: The expression to evaluate (e.g., "2 + 2")
    """
    return eval(expression)
```

The decorator generates a JSON schema from:
- Function name → tool name
- Docstring first line → tool description
- Type hints → parameter types
- Args docstring section → parameter descriptions
- Default values → optional parameters

### Advanced Configuration

#### metadata

Dictionary for LiteLLM tracking and logging. Useful for analytics and debugging.

```python
metadata = {
    "user_id": "user-123",
    "session_id": "session-456",
    "environment": "production"
}
```

LiteLLM includes this metadata in callbacks and logs.

#### cache

Provider-level prompt caching configuration. Reduces latency and cost for repeated prefixes.

```python
cache = True  # Enable default caching
cache = False  # Disable caching
cache = None  # No caching (default)
```

**Advanced configuration** with cache breakpoints:

```python
cache = [
    {"location": "message", "role": "system"},  # Cache system message
    {"location": "message", "index": 0}  # Cache first user message
]
```

Supported by providers like Anthropic. The provider caches specified messages to reuse in subsequent calls.

#### XML Configuration

Control XML tag names for input/output serialization (internal use).

```python
xml_input_root = "input"  # Default
xml_output_root = "output"  # Default
```

You typically don't need to change these. Acorn handles XML serialization internally when communicating with the LLM.

#### max_parse_retries

Number of retry attempts when output validation fails.

```python
max_parse_retries = 2  # Default
max_parse_retries = 5  # More retries
```

When the LLM's output doesn't match the `final_output` schema, acorn automatically retries with an error message. This helps the LLM correct validation errors.

#### stream

Enable streaming responses. Requires implementing the `on_stream` callback.

```python
stream = False  # Default
stream = True   # Enable streaming
```

**Example with streaming:**

```python
class StreamingAgent(Module):
    stream = True
    
    def on_stream(self, chunk):
        """Called for each streamed chunk."""
        if chunk.content:
            print(chunk.content, end="", flush=True)
        if chunk.partial:
            print(f"Partial output: {chunk.partial}")
        if chunk.done:
            print("\nStreaming complete")
```

## Lifecycle Hooks

### on_step

Called after each step in the multi-turn loop. Use this to inspect, log, or modify execution.

```python
def on_step(self, step):
    """Called after each step.
    
    Args:
        step: Step object with execution details
    """
    print(f"Step {step.counter}")
    print(f"Tools called: {[tc.name for tc in step.tool_calls]}")
    
    # Inspect tool results
    for result in step.tool_results:
        if result.error:
            print(f"Error: {result.error}")
        else:
            print(f"Result: {result.output}")
    
    return step
```

**Step object attributes:**

- `step.counter` - Step number (1-indexed)
- `step.model` - Model identifier used
- `step.temperature` - Temperature for this step
- `step.max_tokens` - Max tokens for this step
- `step.tools` - Available tools
- `step.response` - Raw LLM response
- `step.tool_calls` - List of ToolCall objects
- `step.tool_results` - List of ToolResult objects

**Modify execution:**

```python
def on_step(self, step):
    # Add a tool for next iteration
    step.add_tool(new_tool)
    
    # Remove a tool
    step.remove_tool("old_tool")
    
    # Early termination
    if some_condition:
        step.finish(
            answer="Early exit",
            confidence=0.8
        )
    
    return step
```

### on_stream

Called for each chunk when streaming is enabled.

```python
def on_stream(self, chunk):
    """Called for each streamed chunk.
    
    Args:
        chunk: StreamChunk object
    """
    if chunk.content:
        # Text content being streamed
        print(chunk.content, end="", flush=True)
    
    if chunk.partial:
        # Partial structured output
        print(f"Partial: {chunk.partial}")
    
    if chunk.tool_call:
        # Tool call being streamed
        print(f"Tool: {chunk.tool_call}")
    
    if chunk.done:
        # Streaming complete
        print("\nDone")
```

## Module History

The conversation history is accessible as `self.history`. It's a list of message dictionaries:

```python
[
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "...", "content": "..."},
]
```

**Accessing history:**

```python
def on_step(self, step):
    print(f"History length: {len(self.history)}")
    
    for msg in self.history:
        print(f"{msg['role']}: {msg.get('content', '')[:50]}")
    
    return step
```

**Modifying history:**

```python
def on_step(self, step):
    # Append a reminder
    self.history.append({
        "role": "user",
        "content": "Remember to be concise."
    })
    
    # Truncate old messages to manage context length
    if len(self.history) > 50:
        self.history = self.history[:1] + self.history[-40:]
    
    return step
```

## Examples

### Example 1: Text Classifier

Single-turn module that classifies text sentiment.

```python
from pydantic import BaseModel, Field
from acorn import Module

class SentimentClassifier(Module):
    """Classify text sentiment as positive, negative, or neutral."""
    
    class Input(BaseModel):
        text: str = Field(description="Text to classify")
    
    class Output(BaseModel):
        sentiment: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(description="Confidence score 0-1")
    
    initial_input = Input
    final_output = Output
    temperature = 0.3

classifier = SentimentClassifier()
result = classifier(text="I love this product!")
print(f"{result.sentiment} ({result.confidence})")
```

### Example 2: Research Assistant

Multi-turn module with tools and step tracking.

```python
from pydantic import BaseModel, Field
from acorn import Module, tool

class ResearchAssistant(Module):
    """Research topics using multiple tools."""
    
    class Input(BaseModel):
        topic: str = Field(description="Topic to research")
    
    class Output(BaseModel):
        summary: str = Field(description="Research summary")
        sources: list[str] = Field(description="Sources used")
    
    initial_input = Input
    final_output = Output
    max_steps = 5
    
    @tool
    def search(self, query: str) -> list:
        """Search for information.
        
        Args:
            query: Search query
        """
        return search_api(query)
    
    @tool
    def analyze(self, data: str) -> str:
        """Analyze data.
        
        Args:
            data: Data to analyze
        """
        return analysis_result
    
    def on_step(self, step):
        print(f"Step {step.counter}: {[tc.name for tc in step.tool_calls]}")
        return step

assistant = ResearchAssistant()
result = assistant(topic="Climate change")
print(result.summary)
```

## Best Practices

### Choose the Right Mode

Use **single-turn** for:
- Classification tasks
- Simple transformations
- Structured data extraction
- Tasks with all information upfront

Use **multi-turn** for:
- Research and information gathering
- Tasks requiring multiple external calls
- Iterative reasoning
- Complex workflows

### Set Appropriate max_steps

Start with a small number (3-5) and increase if needed. Higher values:
- Increase cost (more LLM calls)
- Increase latency
- May lead to wandering behavior

### Write Clear System Prompts

Good prompts:
- State the agent's role clearly
- Provide specific guidelines
- Include examples when helpful
- Use markdown formatting

```python
class MyAgent(Module):
    """You are a code review assistant.
    
    ## Guidelines
    - Focus on correctness and readability
    - Suggest improvements, don't just criticize
    - Explain your reasoning
    
    ## Example Review
    "The function works but could be more readable..."
    """
```

### Use Descriptive Schemas

Add descriptions to all fields. The LLM uses these to understand what data to provide:

```python
class Output(BaseModel):
    summary: str = Field(description="Concise 2-3 sentence summary")
    key_points: list[str] = Field(description="3-5 main points as bullet items")
    confidence: float = Field(description="Confidence score between 0 and 1")
```

### Handle Errors in Tools

Return error messages as strings rather than raising exceptions:

```python
@tool
def search(self, query: str) -> str:
    """Search for information."""
    try:
        return search_api(query)
    except Exception as e:
        return f"Search failed: {e}"
```

This lets the LLM see the error and potentially recover or try a different approach.

### Monitor with on_step

Use `on_step` to track execution and debug issues:

```python
def on_step(self, step):
    # Log progress
    print(f"Step {step.counter}/{self.max_steps}")
    
    # Check for loops
    tool_names = [tc.name for tc in step.tool_calls]
    if tool_names.count(tool_names[0]) > 3:
        print("Warning: Tool called repeatedly")
    
    return step
```
