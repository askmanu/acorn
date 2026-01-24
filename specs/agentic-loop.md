# Agentic Loop

Acorn modules run in a ReAct-style agentic loop. The model reasons, takes actions via tools, observes results, and continues until completion.

## Loop Flow

```
┌─────────────────────────────────────────────────────────┐
│  1. Module called with initial_input                    │
│     agent(question="...")                               │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│  2. Build initial messages                              │
│     - System prompt                                     │
│     - User message with input data (XML-serialized)     │
│     - __finish__ tool with final_output schema          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
          ┌──────────────────────────────┐
          │  3. Call LLM via LiteLLM     │◄─────────────┐
          └──────────────┬───────────────┘              │
                         │                              │
                         ▼                              │
          ┌──────────────────────────────┐              │
          │  4. Model responds with      │              │
          │     tool call(s)             │              │
          └──────────────┬───────────────┘              │
                         │                              │
                         ▼                              │
                   ┌───────────────┐                    │
                   │ Is __finish__ │                    │
                   │   called?     │                    │
                   └───────┬───────┘                    │
                      yes/ \no                          │
                        /   \                           │
                       ▼     ▼                          │
          ┌────────────────┐  ┌───────────────────┐     │
          │ 5a. Validate   │  │ 5b. Execute       │     │
          │ via Pydantic   │  │ other tools       │     │
          └───────┬────────┘  └─────────┬─────────┘     │
                  │                     │               │
                  ▼                     ▼               │
          ┌────────────────┐   ┌────────────────────┐   │
          │ Return         │   │ 6. on_step         │   │
          │ final_output   │   │    callback        │   │
          └────────────────┘   └─────────┬──────────┘   │
                                         │              │
                                         ▼              │
                                ┌──────────────────┐    │
                                │ max_steps        │    │
                                │ reached?         │    │
                                └────────┬─────────┘    │
                                   yes/ \no             │
                                     /   \              │
                                    ▼     └─────────────┘
                         ┌─────────────────────────┐
                         │ Force __finish__ call   │
                         │ (see Forced Termination)│
                         └────────────┬────────────┘
                                      │
                                      ▼
                         ┌─────────────────────────┐
                         │ Return final_output     │
                         └─────────────────────────┘
```

## Step Lifecycle

Each iteration of the loop is a "step". The step object contains all information about the current state.

### Step Properties

```python
step.counter        # Current step number (1-indexed)
step.model          # Model being used
step.temperature    # Current temperature
step.max_tokens     # Current max_tokens
step.tools          # Currently available tools
step.response       # Model's response this step
step.tool_calls     # List of tool calls made
step.tool_results   # List of tool execution results
```

### Step Methods

```python
step.remove_tool(tool_name)     # Remove a tool for remaining steps
step.add_tool(tool)             # Add a tool for remaining steps
```

---

## Termination

The loop terminates when the model calls `__finish__`. This is the **only** way to produce structured output.

### Normal Termination: __finish__ Tool

The model always returns structured output by calling the `__finish__` tool:

```python
# Model calls:
__finish__(answer="Paris", confidence=0.95, sources=["wiki"])

# Acorn validates via Pydantic, then returns:
AnswerOutput(answer="Paris", confidence=0.95, sources=["wiki"])
```

The flow is always:
1. Model calls `__finish__(field1=..., field2=...)`
2. Tool call arguments (JSON) are validated against `final_output` schema
3. Pydantic model instance is returned

### Forced Termination (max_steps Reached)

When `max_steps` is reached, acorn forces the model to call `__finish__`:

```
max_steps reached?
       │
       ▼
┌─────────────────────────────┐
│ Try tool_choice="__finish__"│
│ (preserves prompt cache)    │
└──────────────┬──────────────┘
               │
        supported?
          /    \
        yes     no
         │       │
         ▼       ▼
┌─────────────┐ ┌─────────────────────────┐
│ Model calls │ │ Append instruction:     │
│ __finish__  │ │ "Call __finish__ now"   │
└─────────────┘ │ + XML parsing fallback  │
                └─────────────────────────┘
```

**Primary strategy**: Use `tool_choice` parameter to force `__finish__` call. This preserves prompt caching since the messages don't change.

**Fallback strategy** (for providers without `tool_choice` support):
1. Append instruction to user message requesting XML output
2. Parse XML response and convert to `__finish__` call internally

```python
class MyAgent(module):
    max_steps = 10  # Enable agentic loop with up to 10 iterations
    # Default is None (single-turn mode)
```

If forced output fails to parse, the `parse_retries` mechanism kicks in - the error is sent back to the model for correction. Only after all retries are exhausted does `ParseError` raise.

### 3. Custom Termination (via on_step)

Force termination by calling `step.finish()` with arguments matching `final_output`:

```python
def on_step(self, step):
    if some_condition:
        # Provide all required fields from final_output schema
        step.finish(answer="...", confidence=0.5, sources=[])
```

To abort with an error, raise an exception:

```python
def on_step(self, step):
    if missing_api_key:
        raise ValueError("Cannot proceed without API key")
```

---

## on_step Callback

Called after each step, after tools have executed but before the next model call. This is where you can inspect state, modify history, and control the next iteration.

### Timing

The `on_step` callback runs at this point in the loop:

```
Model responds with tool calls
       │
       ▼
Tools are executed
       │
       ▼
┌─────────────────────────────────────┐
│  on_step(step) called               │  <-- YOU ARE HERE
│  - step.tool_calls: what was called │
│  - step.tool_results: execution     │
│  - self.history: full conversation  │
└─────────────────────────────────────┘
       │
       ▼
History updated with tool results
       │
       ▼
Next model call (if not finished)
```

### Basic Usage

```python
def on_step(self, step):
    # Inspect state
    print(f"Step {step.counter}")
    print(f"Tool calls: {step.tool_calls}")
    print(f"Tool results: {step.tool_results}")

    # Modify settings for next iteration
    if step.counter > 5:
        step.temperature = 0.1  # Reduce randomness

    # Modify available tools
    if got_enough_data:
        step.remove_tool("search_web")

    return step  # Must return step
```

### Modifying History

The `self.history` list is fully mutable. Add, modify, or remove messages to control what the model sees:

```python
def on_step(self, step):
    # Add a reminder message
    self.history.append({
        "role": "user",
        "content": "Remember to be concise in your next response."
    })

    # Modify the system prompt mid-conversation
    for msg in self.history:
        if msg["role"] == "system":
            msg["content"] += "\n\nNOTE: User prefers bullet points."

    # Trim old messages to manage context length
    if len(self.history) > 50:
        # Keep system message + last 40 messages
        self.history = self.history[:1] + self.history[-40:]

    return step
```

### Modifying Tool Results

The `step.tool_results` list is mutable. Modify results before they're sent to the model:

```python
def on_step(self, step):
    for result in step.tool_results:
        # Truncate large outputs
        if len(str(result.output)) > 10000:
            result.output = str(result.output)[:10000] + "... (truncated)"

        # Add metadata to results
        if result.name == "search_web":
            result.output = {
                "data": result.output,
                "note": "Results may need verification"
            }

    return step
```

### Manual Branching

Spawn branches programmatically when you need custom logic:

```python
def on_step(self, step):
    # Branch when specific conditions are met
    for result in step.tool_results:
        if result.name == "analyze" and result.output.get("needs_review"):
            # Spawn review branch
            review_result = self.branch(ReviewBranch)

            # Inject result into history
            self.history.append({
                "role": "user",
                "content": f"Review complete: {review_result.model_dump_json()}"
            })

    return step
```

### What You Can Do in on_step

| Action | Method |
|--------|--------|
| Change model for next step | `step.model = "..."` |
| Change temperature | `step.temperature = 0.5` |
| Mutate conversation history | `self.history.append(...)`, `self.history = ...` |
| Mutate tool results | `step.tool_results[i].output = ...` |
| Add a tool | `step.add_tool(tool)` |
| Remove a tool | `step.remove_tool(name)` |
| Spawn a branch | `self.branch(Module)` |
| Force completion | `step.finish(**final_output_data)` (kwargs match `final_output` schema) |
| Abort with error | `raise Exception(reason)` |

---

## on_stream Callback

Called for each chunk during streaming responses. The model can return text AND make tool calls in the same response - `on_stream` handles text chunks and partial structured output, while `on_step` handles tool execution results.

```python
def on_stream(self, chunk):
    if chunk.content:
        print(chunk.content, end="", flush=True)
```

### Chunk Properties

```python
chunk.content       # Text content (may be None for tool calls)
chunk.partial       # Partial[T] - partially parsed final_output (if streaming __finish__)
chunk.tool_call     # Partial tool call data (if streaming other tool calls)
chunk.done          # True if this is the final chunk
```

### Partial Streaming for Structured Output

When the model calls `__finish__`, the tool arguments stream as partial JSON. Acorn parses what's available and provides a `Partial[T]` snapshot where `T` is your `final_output` type:

```python
from acorn import module, Partial
from pydantic import BaseModel

class CityInfo(BaseModel):
    city_name: str
    population: int
    explanation: str

class MyAgent(module):
    final_output = CityInfo

    def on_stream(self, chunk):
        # chunk.partial is Partial[CityInfo] - all fields are Optional
        if chunk.partial:
            if chunk.partial.city_name:
                print(f"City: {chunk.partial.city_name}")
            if chunk.partial.population:
                print(f"Pop: {chunk.partial.population}")

        # Also still have text content (chain-of-thought)
        if chunk.content:
            print(chunk.content, end="")
```

**How Partial Works**:
- `Partial[T]` creates a version of `T` where all fields are `Optional`
- As `__finish__` arguments stream, we parse available JSON incrementally
- Fields appear as they're generated by the model
- Useful for UI updates, progress indicators, and early field access

### Streaming + Tool Calls

A single model response can include both reasoning text and tool calls:

```
Model response:
  "Let me search for that information."  <- streamed via on_stream (chunk.content)
  + search_web(query="python creation")  <- tool call, handled after stream
```

The flow:
1. Text chunks streamed via `on_stream` (chain-of-thought reasoning)
2. Tool call arguments streamed (partial data available)
3. For `__finish__`: `chunk.partial` provides `Partial[final_output]` snapshots
4. Tool calls executed after stream completes
5. `on_step` called with both text and tool results

---

## Single-Turn Mode (Default)

By default, modules run in single-turn mode (`max_steps = None`). The model makes one response, calling `__finish__` to return structured output.

```python
from acorn import module
from pydantic import BaseModel, Field

class TextInput(BaseModel):
    text: str

class SentimentOutput(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")

class SentimentClassifier(module):
    """Classify the sentiment of the given text."""

    model = "anthropic/claude-sonnet-4-5-20250514"
    # max_steps = None is the default - single turn

    initial_input = TextInput
    final_output = SentimentOutput
```

### How Single-Turn Works

With `max_steps = None` (default):
- **Single LLM call** - input → `__finish__` call → output
- **Tools allowed** - other tools can be provided, model chooses which to call
- **Output via `__finish__`** - always returns structured data through tool call

```python
# Usage - single LLM call, model calls __finish__ with result
classifier = SentimentClassifier()
result = classifier(text="I love this product!")
print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95
```

### With Tools

Single-turn modules can have other tools. The model chooses: call a tool for information OR call `__finish__` to return.

```python
class ResearchQuery(module):
    """Answer questions, using search if needed."""

    initial_input = QuestionInput
    final_output = AnswerOutput
    tools = [search_web]  # Model can call this, then must call __finish__
    # max_steps = None (default) - single turn, but tools allowed
```

**Note**: In single-turn mode with tools, if the model calls a non-`__finish__` tool, the loop doesn't iterate - the tool executes and the result is returned along with an indication that the model didn't finish. Set `max_steps = N` to allow multi-step tool usage.

### Use Cases

Single-turn modules are ideal for:
- **Classification** - categorizing text into predefined labels
- **Extraction** - pulling structured data from unstructured text
- **Transformation** - rewriting, summarizing, translating
- **Simple Q&A** - questions with straightforward answers
- **Tool dispatch** - single tool call with structured args

---

## XML Fallback for Forced Output

When forcing output at `max_steps` and `tool_choice` isn't supported, acorn falls back to XML-based output extraction.

### Fallback Behavior

Acorn appends an instruction to the user message requesting XML output. Descriptions are placed in `desc` attributes:

```
You must provide your final answer now. Respond with your answer in the following XML structure:
<output>
    <sentiment description="positive, negative, or neutral"></sentiment>
    <confidence description="Confidence score 0-1"></confidence>
</output>

Fill in the values. Do not repeat the descriptions.
```

The model should respond with just the values:

```xml
<output>
    <sentiment>positive</sentiment>
    <confidence>0.95</confidence>
</output>
```

Acorn then parses the XML and converts it internally to a `__finish__` call.

### Why Attributes?

Using `description` as an XML attribute (not content):
- **Clearer separation** - descriptions vs values are distinct
- **Prevents echo** - model won't waste tokens repeating descriptions
- **Easier parsing** - values are always the element content
- **Matches Pydantic** - same attribute name as `Field(description="...")`

### Parse Retries

If the model's output doesn't validate (whether from `__finish__` tool call or XML fallback):
1. Error message sent back to model with the validation failure
2. Model attempts to correct its output
3. After `parse_retries` failures, `ParseError` is raised

```python
class MyAgent(module):
    parse_retries = 2  # Default: 2 retry attempts
```

---

## Error Handling

### Parse Errors

If the `__finish__` arguments don't validate against the `final_output` schema:

1. Validation error sent back to model
2. Model retries with corrected data
3. After `parse_retries` failures, `ParseError` is raised

### API Errors

Handled by LiteLLM's built-in retry logic. Configure via model dict:

```python
model = {
    "model": "anthropic/claude-sonnet-4-5-20250514",
    "num_retries": 3,
    "retry_delay": 1.0,
}
```

### Tool Execution Errors

If a tool raises an exception:

1. Error is caught
2. Error message sent to model as tool result
3. Model can retry or try different approach

```python
@tool
def risky_operation(x: int) -> str:
    if x < 0:
        raise ValueError("x must be positive")
    return "success"

# Model receives:
# Tool error: ValueError: x must be positive
```

---

## Example: Complete Module

```python
from acorn import module, tool
from pydantic import BaseModel, Field

@tool
def search(query: str) -> list[str]:
    """Search for information."""
    return ["result 1", "result 2"]

@tool
def calculate(expression: str) -> float:
    """Evaluate a math expression."""
    return eval(expression)

class QuestionInput(BaseModel):
    question: str

class AnswerOutput(BaseModel):
    answer: str
    confidence: float

class ResearchAgent(module):
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.7
    max_tokens = 4096
    max_steps = 15  # Enable agentic loop (default None = single turn)
    parse_retries = 2

    system_prompt = """
    You are a research assistant. Use tools to find information,
    then call __finish__ with your answer.
    """

    initial_input = QuestionInput
    final_output = AnswerOutput
    tools = [search, calculate]

    def on_step(self, step):
        print(f"Step {step.counter}: {len(step.tool_calls)} tool calls")
        return step

    def on_stream(self, chunk):
        # Stream text content (chain-of-thought)
        if chunk.content:
            print(chunk.content, end="")

        # Stream partial structured output as __finish__ arguments arrive
        if chunk.partial and chunk.partial.answer:
            print(f"\n[Partial answer: {chunk.partial.answer[:50]}...]")

# Usage
agent = ResearchAgent()
result = agent(question="What is 2+2 and what year was Python created?")
print(result.answer)       # "4, and Python was created in 1991"
print(result.confidence)   # 0.95
```
