# Acorn - Context for Claude Code

## What is Acorn?
LLM agent framework with structured I/O. Built on Pydantic for schemas and LiteLLM for multi-provider access.

**Status**: v0.1.0 - Production-ready for single-turn and multi-turn agentic loops. 128 tests, 90% coverage.

## Core Architecture

### Module (First-Class Citizen)
- Base class for all agents: `class MyAgent(module)`
- All config lives in class attributes: `model`, `temperature`, `max_steps`, `tools`, etc.
- Invoked like a function: `agent(question="...")` returns structured Pydantic output

### Structured I/O
- **User-facing**: Pydantic models for `initial_input` and `final_output`
- **LLM-facing**: XML for data serialization (internal), native tool calling APIs
- **Output mechanism**: ALWAYS via `__finish__` tool call (no direct XML output mode)
- Single-turn mode (`max_steps = None`): input → `__finish__` → output
- Multi-turn mode (`max_steps = N`): ReAct loop with tools

### Tools
- Defined with `@tool` decorator on functions or methods
- Schemas auto-generated from type hints and docstrings
- `__finish__` tool auto-added to every module for structured output
- Tools use native API formats (JSON), NOT XML

### Agentic Loop
- ReAct-style: reason → act (tool calls) → observe → repeat
- `on_step(step)` callback after each step for inspection/modification
- `on_stream(chunk)` callback for streaming text and partial structured output
- Termination: model calls `__finish__` with data matching `final_output` schema
- Forced termination at `max_steps`: uses `tool_choice="__finish__"` (or XML fallback)
- Parse error recovery: validation failures sent back to model for retry

### Branching (Extend, Not Replace)
- Branches EXTEND parent modules, not replace them
- Inherit: system prompt (appended), tools (added), full history
- No `initial_input` - context from inherited history
- Must define `final_output` for structured return
- Registered in `branches = {"name": BranchModule}`
- Auto-generates parameter-less tools: `fact_check() -> FactCheckOutput`

## Key Design Decisions

1. **XML for Data, Not Prompts**: XML only for structured I/O with LLM (internal). System prompts are Markdown. Users never write XML.

2. **Module Docstring = System Prompt**: Class docstring becomes system prompt (cleaner API). For branches, docstring is appended to parent's prompt.

3. **`__finish__` is the ONLY Output Path**: Structured output always via tool call. XML output only as fallback when `tool_choice` unsupported.

4. **History is Mutable**: Full access to `self.history` in `on_step` - add/modify/remove messages.

5. **Tool Results are Mutable**: Modify `step.tool_results` before returning from `on_step` to change what model sees.

6. **Model Can Be Dict**: `model` accepts string or dict for advanced LiteLLM config.

## Implementation Status (v0.1.0)

### ✅ Completed
- Single-turn and multi-turn execution
- Tool calling with auto-schema generation
- Parse error recovery with retries
- Dynamic tool management (add/remove in `on_step`)
- Step callbacks and streaming
- Forced termination at `max_steps` (tool_choice + XML fallback)
- Branching (declarative and manual)
- Streaming responses with partial structured outputs (Phase 8)
- Provider caching (Phase 10)

### 📋 Planned
- Async support (v0.2)

## File Structure
- `acorn/` - Source code
- `specs/` - Detailed design specifications (see below)
- `examples/` - Working examples
- `tests/` - Test suite (128 tests, 90% coverage)
- `IMPLEMENTATION_STATUS.md` - Detailed feature status and API reference
- `PROGRESS_SUMMARY.md` - Development progress

## Spec Files (Full Details)
- `intro.md` - Philosophy, tech stack, v0.1 scope
- `core-concepts.md` - Module, tools, I/O schemas, system prompt
- `agentic-loop.md` - ReAct loop, step lifecycle, callbacks, termination
- `branching.md` - Branch inheritance model, nested branches
- `data-serialization.md` - XML serialization details, Pydantic ↔ XML
- `api-reference.md` - Complete API surface

## Common Patterns

### Single-Turn Module
```python
class Classifier(module):
    """Classify sentiment of text."""
    initial_input = TextInput
    final_output = SentimentOutput
    # max_steps = None (default)
```

### Multi-Turn Agent with Tools
```python
class ResearchAgent(module):
    """Research assistant."""
    max_steps = 15
    tools = [search_web, calculate]
    final_output = AnswerOutput

    def on_step(self, step):
        # Inspect, modify history, manage tools
        return step
```

### Branch Module (Extends Parent)
```python
class FactCheckBranch(module):
    """Verify claims discussed in conversation."""
    # No initial_input - uses inherited history
    tools = [deep_search]  # Added to parent's tools
    final_output = VerificationOutput
```

## Remember
- NEVER create XML manually - use Pydantic models
- Output ALWAYS via `__finish__` tool call
- Branches inherit and extend, they don't replace
- History and tool results are mutable in `on_step`
- `max_steps = None` is single-turn (still allows tools)
- Model can be string or dict for advanced config
