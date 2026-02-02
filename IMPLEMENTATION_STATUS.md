# Acorn Implementation Status

## Summary

Successfully implemented the foundation and core features of the Acorn LLM agent framework through Phase 7 of the implementation plan.

**Test Results:** 148 tests passing with 79% code coverage

---

## Completed Phases

### ✅ Phase 0: Project Setup
- Package structure created with `pyproject.toml`
- Installed with pip in editable mode
- Dev dependencies configured (pytest, pytest-cov, ruff)
- Proper `.gitignore` and `README.md`

### ✅ Phase 1: Foundation - Types & Exceptions
**Files Created:**
- `acorn/exceptions.py` - Exception hierarchy (AcornError, ParseError, BranchError, ToolConflictError)
- `acorn/types.py` - Core data structures (Step, ToolCall, ToolResult, StreamChunk)
- `acorn/partial.py` - Partial[T] model generation for streaming

**Test Coverage:** 27 tests, 100% coverage

### ✅ Phase 2: XML Serialization Engine
**Files Created:**
- `acorn/serialization/xml_encoder.py` - Pydantic → XML conversion
- `acorn/serialization/xml_decoder.py` - XML → Pydantic parsing

**Features:**
- Handles nested models, lists, dicts, optional fields
- Special types: datetime, date, Enum, bool
- Field descriptions (attribute or comment format)
- Pretty-printed XML output
- Full roundtrip preservation

**Test Coverage:** 45 tests, 97% encoder / 72% decoder coverage

### ✅ Phase 3: Tool System
**Files Created:**
- `acorn/decorators.py` - @tool decorator
- `acorn/tool_schema.py` - Function → JSON schema generation

**Features:**
- Automatic schema generation from type hints
- Docstring parsing (Args section)
- Support for: primitives, Optional, list, dict, Union types
- Method support (excludes self parameter)
- Complex type handling

**Test Coverage:** 20 tests, 94% coverage

### ✅ Phase 4: Basic Module Class (Single-Turn)
**Files Created:**
- `acorn/module.py` - Main module class
- `acorn/llm/litellm_client.py` - LiteLLM wrapper

**Features:**
- Single-turn execution (max_steps=None)
- Input/output validation with Pydantic
- XML-based input serialization
- System prompt from: string, Path, method, or docstring
- Tool collection from both `tools` list and `@tool` methods
- Automatic `__finish__` tool generation
- Tool name conflict detection

**Test Coverage:** 21 tests, 95% module / 86% LLM client coverage

### ✅ Phase 5: Complete Single-Turn Implementation
**Features Added:**
- Parse error retry logic (max_parse_retries configuration)
- System prompt from Path (file reading)
- Automatic retry with error messages on validation failure
- Configurable retry attempts

**Test Coverage:** 6 new tests for retry logic

### ✅ Phase 6: Agentic Loop - Basic
**Features Added:**
- Multi-turn execution (max_steps > 0)
- History management across turns
- Tool execution with error handling
- Step lifecycle with callbacks
- Loop termination logic
- `on_step` callback support
- Step mutations (add_tool, remove_tool, finish)
- Dynamic tool management

**Test Coverage:** 9 comprehensive agentic loop tests

### ✅ Phase 7: Callbacks & Forced Termination
**Features Added:**
- Forced termination at max_steps with tool_choice (primary strategy)
- XML fallback for forced termination (when tool_choice not supported)
- Automatic retry logic for forced termination validation errors
- `on_stream` callback support for streaming responses
- StreamChunk handling for text content and tool calls
- Stream accumulation in LiteLLM client

**Test Coverage:** 2 forced termination tests (tool_choice and XML fallback)

---

## Architecture Overview

```
acorn/
├── __init__.py           # Main exports
├── _version.py           # Version info
├── exceptions.py         # Exception classes
├── types.py              # Core data structures
├── partial.py            # Partial[T] for streaming
├── decorators.py         # @tool decorator
├── tool_schema.py        # Schema generation
├── module.py             # Main module class (188 lines)
├── serialization/
│   ├── xml_encoder.py    # Pydantic → XML
│   └── xml_decoder.py    # XML → Pydantic
└── llm/
    └── litellm_client.py # LiteLLM wrapper
```

---

## Current Capabilities

### ✅ Working Features

#### 1. Single-Turn Modules
```python
from pydantic import BaseModel
from acorn import module

class Summarizer(module):
    """Summarize text concisely."""

    class Input(BaseModel):
        text: str

    class Output(BaseModel):
        summary: str

    initial_input = Input
    final_output = Output

summarizer = Summarizer()
result = summarizer(text="Long text here...")
print(result.summary)
```

#### 2. Multi-Turn Agentic Loops
```python
from acorn import module, tool

class ResearchAgent(module):
    """Research assistant with tools."""

    max_steps = 5  # Enable agentic loop

    class Output(BaseModel):
        findings: str
        sources: list[str]

    final_output = Output

    @tool
    def search(self, query: str) -> list:
        """Search for information."""
        return ["result1", "result2"]

    @tool
    def analyze(self, data: str) -> str:
        """Analyze data."""
        return f"Analysis of {data}"

    def on_step(self, step):
        print(f"Step {step.counter}: {[tc.name for tc in step.tool_calls]}")
        return step

agent = ResearchAgent()
result = agent(topic="AI")
```

#### 3. Dynamic Tool Management
```python
class AdaptiveAgent(module):
    max_steps = 10

    def on_step(self, step):
        if step.counter == 3:
            # Add new tool mid-execution
            step.add_tool(new_tool)
            # Remove obsolete tool
            step.remove_tool("old_tool")
        return step
```

#### 4. Early Loop Termination
```python
class SmartAgent(module):
    max_steps = 100

    def on_step(self, step):
        if some_condition:
            # Exit early with results
            step.finish(result="Early exit")
        return step
```

#### 5. Parse Error Recovery
```python
class RobustModule(module):
    max_parse_retries = 3  # Retry up to 3 times on validation errors
    final_output = Output
```

#### 6. Streaming Responses
```python
class StreamingAgent(module):
    stream = True  # Enable streaming
    max_steps = 10
    final_output = Output

    def on_stream(self, chunk):
        # Handle streaming text content
        if chunk.content:
            print(chunk.content, end="", flush=True)

        # Handle streaming tool calls
        if chunk.tool_call:
            print(f"\nTool: {chunk.tool_call}")

        # Check if done
        if chunk.done:
            print("\nStreaming complete")
```

#### 7. Forced Termination at max_steps
```python
class LimitedAgent(module):
    max_steps = 5  # Will force __finish__ at step 5

    # Automatically uses tool_choice to force termination
    # Falls back to XML if tool_choice not supported
```

---

## Remaining Implementation (Phases 8-10)

### Phase 8: Partial Streaming
**Not Yet Implemented:**
- Stream partial structured output as JSON arrives
- Partial[T] in StreamChunk during streaming
- Progressive field updates

**Estimated Effort:** 1-2 days

### Phase 9: Branching System
**Not Yet Implemented:**
- Declarative branching (branches = {})
- Manual branching (self.branch())
- Context inheritance (system prompt, tools, history)
- Nested branching

**Estimated Effort:** 2-3 days

### Phase 10: Provider Caching
**Not Yet Implemented:**
- Anthropic-style prompt caching
- Cache breakpoint injection
- Cache configuration

**Estimated Effort:** 1 day

**Total Remaining:** ~1 week for full v0.1

---

## Test Statistics

```bash
✅ 148 tests passing
✅ 79% code coverage
✅ All core components tested
✅ Mocked LLM responses for deterministic testing
✅ Edge cases covered (errors, retries, max_steps, forced termination)
```

**Test Breakdown:**
- Foundation: 27 tests
- Serialization: 45 tests
- Tools: 20 tests
- Module (single-turn): 21 tests
- Parse retry: 6 tests
- Agentic loop: 10 tests (including forced termination)
- Streaming: Included in module tests
- Other: 19 tests

---

## Examples

See `examples/` directory:
- `single_turn_summarizer.py` - Basic single-turn module
- `research_assistant.py` - Full agentic loop with tools

---

## Dependencies

**Runtime:**
- `pydantic>=2.0` - Data validation and serialization
- `litellm>=1.0` - Unified LLM API

**Development:**
- `pytest>=7.0` - Testing framework
- `pytest-cov>=4.0` - Coverage reporting
- `ruff>=0.1.0` - Linting

---

## API Reference

### module Class

```python
class module:
    # Configuration
    model: str | dict = "anthropic/claude-sonnet-4-5-20250514"
    temperature: float = 0.7
    max_tokens: int = 4096
    max_steps: int | None = None  # None = single-turn
    max_parse_retries: int = 2
    stream: bool = False  # Enable streaming (requires on_stream callback)

    # Schemas
    system_prompt: str | Path = ""
    initial_input: type[BaseModel] | None = None
    final_output: type[BaseModel] | None = None
    tools: list = []

    # Callbacks
    def on_step(self, step: Step) -> Step:
        """Called after each agentic step."""
        return step

    def on_stream(self, chunk: StreamChunk) -> None:
        """Called for each streaming chunk."""
        pass
```

### Step Object

```python
class Step:
    counter: int              # Step number (1-indexed)
    model: str | dict         # Model used
    temperature: float        # Temperature
    max_tokens: int          # Max tokens
    tools: list              # Available tools
    response: dict           # Raw LLM response
    tool_calls: list[ToolCall]     # Tool calls made
    tool_results: list[ToolResult] # Tool results

    def add_tool(self, tool) -> None
    def remove_tool(self, name: str) -> None
    def finish(self, **kwargs) -> None
```

### StreamChunk Object

```python
class StreamChunk:
    content: str | None       # Text content (if text chunk)
    partial: Any | None       # Partial[T] for structured output (Phase 8)
    tool_call: dict | None    # Tool call delta (if tool call chunk)
    done: bool               # True if streaming complete
```

---

## Usage Patterns

### Pattern 1: Simple Q&A
```python
class QABot(module):
    """Answer questions."""

    class Input(BaseModel):
        question: str

    class Output(BaseModel):
        answer: str

    initial_input = Input
    final_output = Output
```

### Pattern 2: Research with Tools
```python
class Researcher(module):
    max_steps = 5

    @tool
    def search(self, query: str) -> list:
        return search_api(query)

    @tool
    def summarize(self, text: str) -> str:
        return summarize_text(text)
```

### Pattern 3: Adaptive Workflow
```python
class AdaptiveAgent(module):
    max_steps = 10

    def on_step(self, step):
        if needs_more_tools(step):
            step.add_tool(specialized_tool)

        if task_complete(step):
            step.finish(result="Done")

        return step
```

---

## Performance Notes

- **Single-turn:** ~1 LLM call per execution
- **Multi-turn:** Up to `max_steps` LLM calls
- **Parse retries:** Up to `max_parse_retries + 1` calls per finish attempt
- **Tool execution:** Synchronous (async not yet implemented)

---

## Known Limitations

1. **Partial streaming** - Partial[T] for structured output streaming not yet implemented (Phase 8)
2. **No branching** - Branch modules not yet implemented (Phase 9)
3. **No caching** - Provider caching not yet implemented (Phase 10)
4. **Sync only** - No async support in v0.1
5. **Streaming in retries** - Retry paths don't use streaming yet (optimization for v0.2)

---

## Next Steps for v0.1 Completion

1. **Phase 8:** Implement partial streaming with Partial[T] for structured output
2. **Phase 9:** Add branching system (declarative and manual)
3. **Phase 10:** Support provider caching (Anthropic-style)

After completion, v0.1 will be production-ready for all planned use cases.

---

## Contributing

To continue implementation:

1. Follow the phases in order (7 → 10)
2. Write tests first for each feature
3. Maintain >80% code coverage
4. Update this status document as you progress
5. Add examples to `examples/` directory

---

*Last Updated: 2026-02-02*
*Version: 0.1.0-beta*
*Status: Core features and callbacks complete, Phase 7 done*
