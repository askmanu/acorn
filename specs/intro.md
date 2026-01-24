# Acorn

A Python library for building LLM agents with structured inputs/outputs.

## Philosophy

- **Module as first-class citizen**: All configuration lives in the module class
- **Pydantic for schemas**: Users define inputs/outputs with Pydantic, never write XML
- **XML for LLM communication**: Structured data serialized to XML internally
- **Native tool calling**: Tools use provider APIs directly (not XML)
- **ReAct-style loops**: Agents work iteratively with tool calls until completion
- **Branching**: Spawn sub-agents for specialized tasks

## Tech Stack

- **LiteLLM**: Backend for multi-provider LLM access
- **Pydantic**: Schema definition and validation

## Quick Example

```python
from acorn import module, tool
from pydantic import BaseModel, Field

@tool
def search(query: str) -> list[str]:
    """Search for information."""
    return ["result1", "result2"]

class QuestionInput(BaseModel):
    question: str

class AnswerOutput(BaseModel):
    answer: str
    confidence: float

class ResearchAgent(module):
    model = "anthropic/claude-sonnet-4-5-20250514"
    system_prompt = "You are a research assistant. Use tools to find information."

    initial_input = QuestionInput
    final_output = AnswerOutput
    tools = [search]

# Usage
agent = ResearchAgent()
result = agent(question="What is Python?")
print(result.answer)
```

## Specs

- [Core Concepts](./core-concepts.md) - Module, tools, inputs/outputs, system prompt
- [Agentic Loop](./agentic-loop.md) - ReAct loop, step lifecycle, callbacks, termination
- [Branching](./branching.md) - Sub-agents, declarative and manual branching
- [Data Serialization](./data-serialization.md) - Pydantic to XML, internal format
- [API Reference](./api-reference.md) - Complete API surface

## v0.1 Scope

**Included:**
- Module class with LiteLLM backend
- Pydantic input/output schemas
- Tool decorator with auto schema generation
- ReAct agentic loop
- `__finish__` tool for structured output (only output mechanism)
- `on_step` and `on_stream` callbacks
- `Partial[T]` streaming for structured output
- Provider-level caching (Anthropic-style)
- Forced termination via `tool_choice` (with XML fallback)
- Parse error retry with feedback
- XML serialization for inputs (internal)

**Not included (future):**
- Memory abstractions
- Prompt optimization
- Static context injection
- Parallel branches
- Branch context inheritance
