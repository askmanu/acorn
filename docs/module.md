# Module

The `Module` class is the foundation of acorn. It represents an LLM agent with structured inputs, outputs, and tool-calling capabilities.

## What is Module?

A Module encapsulates everything needed to run an LLM agent:
- Model configuration (which LLM to use, temperature, etc.)
- Input and output schemas (using Pydantic models)
- Tools the agent can call
- Execution mode (single-turn or multi-turn agentic loop)
- Lifecycle hooks for customization

Think of a Module as a reusable, configurable LLM workflow with type safety built in.

## Basic Usage

Create a Module by subclassing and defining schemas:

```python
from pydantic import BaseModel, Field
from acorn import Module

class Input(BaseModel):
    text: str = Field(description="Text to analyze")

class Output(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")

class SentimentAnalyzer(Module):
    """Analyze the sentiment of text."""
    
    initial_input = Input
    final_output = Output

# Use it
analyzer = SentimentAnalyzer()
result = analyzer(text="I love this!")

print(result.sentiment)  # "positive"
print(result.confidence)  # 0.95
```

The Module validates inputs, calls the LLM, and returns validated output as a Pydantic model.

## Execution Modes

Modules run in two modes depending on `max_steps`:

### Single-Turn Mode

Default behavior when `max_steps` is `None`. The Module makes one LLM call and returns structured output immediately.

```python
class Translator(Module):
    """Translate text to French."""
    
    initial_input = Input
    final_output = Output
    # max_steps = None (default)
```

**Flow:**
1. Validate input against `initial_input` schema
2. Call LLM with input and tools (including `__finish__`)
3. LLM calls `__finish__` with structured output
4. Validate output against `final_output` schema
5. Return Pydantic model instance

### Multi-Turn Mode

Enable with `max_steps = N` for agentic loops where the LLM can call tools iteratively.

```python
class ResearchAgent(Module):
    """Research assistant that uses tools."""
    
    final_output = Output
    max_steps = 5  # Up to 5 iterations
    
    @tool
    def search(self, query: str) -> list:
        """Search for information."""
        return search_api(query)
```

**Flow:**
1. Validate input
2. Loop up to N times:
   - Call LLM with current history and tools
   - Execute any tool calls
   - Add results to history
   - If `__finish__` called, validate and return output
3. If max steps reached, force termination

## Configuration Attributes

### Model Settings

Control which LLM to use and how it behaves:

```python
class MyModule(Module):
    model = "anthropic/claude-sonnet-4-5-20250514"  # LiteLLM identifier
    temperature = 0.7  # Sampling temperature (0-2)
    max_tokens = 4096  # Maximum response length
```

You can also use a dict for advanced configuration:

```python
model = {
    "id": "anthropic/claude-sonnet-4-5-20250514",
    "vertex_location": "us-central1",
    "vertex_credentials": "path/to/creds.json",
    "reasoning": True  # Enable extended thinking
}
```

The `reasoning` parameter enables extended thinking modes:
- `True` - Use model's default reasoning
- `"low"`, `"medium"`, `"high"` - Specific reasoning levels

### System Prompt

Provide instructions to the LLM. Multiple options:

**Docstring (recommended):**
```python
class Summarizer(Module):
    """Summarize text concisely.
    
    Keep summaries under 100 words.
    Focus on key points.
    """
```

**String attribute:**
```python
class Summarizer(Module):
    system_prompt = "Summarize text concisely. Keep under 100 words."
```

**File path:**
```python
from pathlib import Path

class Summarizer(Module):
    system_prompt = Path("prompts/summarizer.md")
```

**Dynamic method:**
```python
from datetime import date

class Assistant(Module):
    def system_prompt(self):
        return f"You are an assistant. Today is {date.today()}."
```

### Schemas

Define input and output structure using Pydantic models:

```python
class QuestionInput(BaseModel):
    question: str = Field(description="The question to answer")
    context: str | None = Field(default=None, description="Optional context")

class AnswerOutput(BaseModel):
    answer: str = Field(description="The answer")
    sources: list[str] = Field(description="Sources used")

class QA(Module):
    initial_input = QuestionInput  # What the module accepts
    final_output = AnswerOutput    # What it returns
```

**Rules:**
- `initial_input` is optional (can call with no arguments)
- `final_output` is required for single-turn mode
- `final_output` is optional for multi-turn mode (can return `None`)

### Tools

Tools are functions the LLM can call. Define them with the `@tool` decorator:

```python
from acorn import tool

@tool
def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Search the web for information.
    
    Args:
        query: The search query
        max_results: Maximum results to return
    """
    return search_api(query, max_results)

class Agent(Module):
    tools = [search_web]
```

**As methods:**
```python
class Agent(Module):
    @tool
    def search_web(self, query: str) -> list[dict]:
        """Search the web."""
        # Has access to self
        return self.search_api(query)
```

Tool schemas are auto-generated from:
- Function name → tool name
- Docstring first line → description
- Type hints → parameter types
- Docstring Args section → parameter descriptions
- Default values → optional parameters

### Advanced Settings

**Parse retries:**
```python
max_parse_retries = 2  # Retry validation failures
```

When the LLM's output doesn't match the schema, acorn retries with an error message.

**Streaming:**
```python
stream = True  # Enable streaming responses

def on_stream(self, chunk):
    if chunk.content:
        print(chunk.content, end="")
```

**Metadata:**
```python
metadata = {"user_id": "123", "session": "abc"}  # LiteLLM tracking
```

**Caching:**
```python
cache = True  # Enable prompt caching (provider-dependent)

# Or custom cache points:
cache = [
    {"location": "message", "role": "system"},
    {"location": "message", "index": 0}
]
```

**XML configuration:**
```python
xml_input_root = "input"    # Root tag for input XML
xml_output_root = "output"  # Root tag for output XML
```

## Lifecycle Hooks

Customize behavior at key points in execution:

### on_step

Called after each step in multi-turn mode. Inspect and modify the step:

```python
def on_step(self, step):
    print(f"Step {step.counter}")
    print(f"Tools called: {[tc.name for tc in step.tool_calls]}")
    
    # Modify tool results before next call
    for result in step.tool_results:
        if len(str(result.output)) > 1000:
            result.output = str(result.output)[:1000] + "..."
    
    # Add/remove tools dynamically
    if step.counter > 3:
        step.remove_tool("expensive_tool")
    
    # Early termination
    if some_condition:
        step.finish(answer="Done early", sources=[])
    
    # Adjust parameters for next iteration
    step.temperature = 0.5
    
    return step
```

**What you can do:**
- Inspect tool calls and results
- Modify tool results (affects what LLM sees)
- Add or remove tools
- Change model parameters
- Force early termination with `step.finish()`
- Access and modify conversation history via `self.history`

### on_stream

Called during streaming responses (when `stream = True`):

```python
def on_stream(self, chunk):
    # Text content (chain-of-thought)
    if chunk.content:
        print(chunk.content, end="", flush=True)
    
    # Partial structured output
    if chunk.partial:
        if chunk.partial.answer:
            print(f"\nAnswer: {chunk.partial.answer}")
    
    # Completion
    if chunk.done:
        print("\nDone!")
```

## History

Access the full conversation history:

```python
def on_step(self, step):
    # Read history
    for msg in self.history:
        print(f"{msg['role']}: {msg.get('content', '')[:50]}")
    
    # Modify history (affects next LLM call)
    self.history.append({
        "role": "user",
        "content": "Remember to cite sources."
    })
    
    # Trim old messages
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

## Examples

### Example 1: Single-Turn Classifier

```python
from pydantic import BaseModel, Field
from acorn import Module

class Input(BaseModel):
    text: str

class Output(BaseModel):
    category: str = Field(description="news, blog, or social")
    confidence: float

class Classifier(Module):
    """Classify text into categories."""
    
    initial_input = Input
    final_output = Output
    temperature = 0.3

classifier = Classifier()
result = classifier(text="Breaking: New discovery announced...")
print(result.category)  # "news"
```

### Example 2: Multi-Turn Research Agent

```python
from acorn import Module, tool
from pydantic import BaseModel, Field

class Output(BaseModel):
    summary: str
    key_facts: list[str]
    sources: list[str]

class Researcher(Module):
    """Research topics using available tools."""
    
    final_output = Output
    max_steps = 10
    
    @tool
    def search(self, query: str) -> list[str]:
        """Search for information."""
        return ["fact1", "fact2", "fact3"]
    
    @tool
    def analyze(self, text: str) -> dict:
        """Analyze text for key information."""
        return {"insights": ["insight1", "insight2"]}
    
    def on_step(self, step):
        print(f"Step {step.counter}: {[tc.name for tc in step.tool_calls]}")
        return step

researcher = Researcher()
result = researcher()
print(result.summary)
```

### Example 3: Dynamic Tool Management

```python
class AdaptiveAgent(Module):
    """Agent that adapts tools based on progress."""
    
    final_output = Output
    max_steps = 15
    
    tools = [basic_search, calculate]
    
    def on_step(self, step):
        # Add advanced tools after initial exploration
        if step.counter == 3:
            step.add_tool(advanced_search)
            step.add_tool(data_analysis)
        
        # Remove expensive tools near the end
        if step.counter >= 12:
            step.remove_tool("advanced_search")
            step.temperature = 0.3  # More focused
        
        return step
```

### Example 4: History Management

```python
class ConversationAgent(Module):
    """Agent with conversation memory management."""
    
    final_output = Output
    max_steps = 20
    
    def on_step(self, step):
        # Keep only recent history to manage context length
        if len(self.history) > 30:
            # Keep system prompt + last 25 messages
            system_msg = self.history[0]
            recent = self.history[-25:]
            self.history = [system_msg] + recent
        
        # Add periodic reminders
        if step.counter % 5 == 0:
            self.history.append({
                "role": "user",
                "content": "Remember to stay focused on the main topic."
            })
        
        return step
```

## Common Patterns

### Validation and Error Handling

```python
from acorn import ParseError

try:
    result = module(input_data="...")
except ParseError as e:
    print(f"Output validation failed: {e}")
    print(f"Raw output: {e.raw_output}")
```

### No Input Schema

For modules that don't need structured input:

```python
class SimpleAgent(Module):
    """Agent with no input schema."""
    
    # No initial_input defined
    final_output = Output
    max_steps = 5

agent = SimpleAgent()
result = agent()  # No arguments needed
```

### No Output Schema (Multi-Turn Only)

For side-effect-focused agents:

```python
class TaskExecutor(Module):
    """Execute tasks without returning structured output."""
    
    max_steps = 10
    final_output = None  # No structured output
    
    @tool
    def execute_task(self, task: str) -> str:
        """Execute a task."""
        perform_action(task)
        return "Done"

executor = TaskExecutor()
result = executor()  # Returns None after executing tools
assert result is None
```

### Conditional Early Exit

```python
def on_step(self, step):
    # Check if we have enough information
    search_count = sum(1 for tc in step.tool_calls if tc.name == "search")
    
    if search_count >= 3:
        # We've searched enough, finish now
        step.finish(
            answer="Based on 3 searches...",
            sources=["source1", "source2", "source3"]
        )
    
    return step
```

## Best Practices

**Use descriptive field descriptions:**
```python
class Output(BaseModel):
    answer: str = Field(description="Clear, concise answer to the question")
    confidence: float = Field(description="Confidence score between 0 and 1")
```

Good descriptions help the LLM understand what you want.

**Set appropriate max_steps:**
- Single-turn: `max_steps = None` (default)
- Simple multi-turn: `max_steps = 5`
- Complex workflows: `max_steps = 15-20`

**Use lower temperature for structured tasks:**
```python
class DataExtractor(Module):
    temperature = 0.3  # More deterministic
```

**Manage context length in long conversations:**
```python
def on_step(self, step):
    if len(self.history) > 50:
        self.history = self.history[:1] + self.history[-40:]
    return step
```

**Provide clear tool descriptions:**
```python
@tool
def search(query: str, limit: int = 5) -> list:
    """Search the knowledge base for relevant documents.
    
    Args:
        query: Natural language search query
        limit: Maximum number of documents to return
    """
```

**Use on_step for debugging:**
```python
def on_step(self, step):
    print(f"\n=== Step {step.counter} ===")
    for tc in step.tool_calls:
        print(f"Tool: {tc.name}")
        print(f"Args: {tc.arguments}")
    for tr in step.tool_results:
        print(f"Result: {tr.output}")
    return step
```

## Next Steps

- Read [Getting Started](getting-started.md) for complete examples
- Check the API reference in `specs/api-reference.md` for detailed attribute documentation
- See `specs/core-concepts.md` for deeper conceptual explanations
