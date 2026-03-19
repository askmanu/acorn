# Acorn - Context for Claude Code

## What is Acorn?
LLM agent framework with structured I/O. Built on Pydantic for schemas and LiteLLM for multi-provider access.

## Core Architecture

### Module (First-Class Citizen)
- Base class for all agents: `class MyAgent(module)`
- All config lives in class attributes: `model`, `temperature`, `max_steps`, `tools`, etc.
- **Async-first**: `__call__` is async, `run()` provides sync access
  - `result = await agent(question="...")` — async
  - `result = agent.run(question="...")` — sync wrapper

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

7. **Async-First**: `__call__` is async. `run()` provides sync access via `asyncio.run()` or a background thread when already in an event loop.

## File Structure
- `acorn/` - Source code
- `examples/` - Working examples
- `tests/` - Test suite

## Common Patterns

### Single-Turn Module
```python
class Classifier(module):
    """Classify sentiment of text."""
    initial_input = TextInput
    final_output = SentimentOutput
    # max_steps = None (default)

result = await Classifier()(text="Hello world")
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

result = await ResearchAgent()(topic="AI")
# Or sync: ResearchAgent().run(topic="AI")
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
- `__call__` is async; use `run()` for sync access
