# Acorn 🌰

**LLM agent framework with structured I/O, heavily influenced by DSPy.**

Build AI agents with type-safe inputs and outputs, automatic tool calling, and powerful agentic loops.

[![Tests](https://img.shields.io/badge/tests-128%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.10+-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

---

## ✨ Features

- 🎯 **Structured I/O** - Pydantic models for inputs and outputs
- 🤖 **Agentic Loops** - Multi-turn execution with tool calling
- 🛠️ **Auto Tool Schemas** - Generate from type hints and docstrings
- 🔄 **Dynamic Tools** - Add/remove tools during execution
- ✅ **Parse Error Recovery** - Automatic retry on validation failures
- 📊 **Step Callbacks** - Full control over loop behavior
- 🔌 **LiteLLM Integration** - Works with any LLM provider

---

## 🚀 Quick Start

### Installation

```bash
# Install from source (PyPI package coming soon)
git clone <repo>
cd acorn
pip install -e .
```

### Single-Turn Example

```python
from pydantic import BaseModel
from acorn import module

class Summarizer(module):
    """Summarize text concisely."""

    class Input(BaseModel):
        text: str
        max_words: int = 100

    class Output(BaseModel):
        summary: str
        word_count: int

    initial_input = Input
    final_output = Output

# Use it
summarizer = Summarizer()
result = summarizer(
    text="Long article text here...",
    max_words=50
)

print(result.summary)
print(f"Words: {result.word_count}")
```

### Multi-Turn Agentic Loop

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
        # Your search implementation
        return ["result1", "result2"]

    @tool
    def analyze(self, data: str) -> str:
        """Analyze collected data."""
        # Your analysis implementation
        return f"Analysis: {data}"

    def on_step(self, step):
        """Called after each step."""
        print(f"Step {step.counter}")

        # Early termination if done
        if len(step.tool_results) >= 3:
            step.finish(
                findings="Sufficient data collected",
                sources=["source1", "source2"]
            )

        return step

# Use it
agent = ResearchAgent()
result = agent()
```

---

## 📚 Core Concepts

### Module
Base class for LLM agents. Configure with:
- `model` - LLM to use (default: Claude Sonnet 4.5)
- `temperature` - Sampling temperature
- `max_steps` - Max agentic loop iterations (None = single-turn)
- `initial_input` - Pydantic model for input schema
- `final_output` - Pydantic model for output schema
- `tools` - List of available tools

### Tools
Functions the LLM can call:

```python
@tool
def search(query: str, limit: int = 10) -> list:
    """Search for information.

    Args:
        query: The search query
        limit: Maximum results to return
    """
    return search_api(query, limit)
```

Schema is automatically generated from type hints and docstring.

### Step Callback
Control agentic loop execution:

```python
def on_step(self, step):
    # Access step info
    print(f"Step {step.counter}")
    print(f"Tools called: {[tc.name for tc in step.tool_calls]}")

    # Dynamic tool management
    step.add_tool(new_tool)
    step.remove_tool("old_tool")

    # Early termination
    if condition:
        step.finish(result="Early exit")

    return step
```

---

## 🎯 Examples

See `examples/` directory:
- [`single_turn_summarizer.py`](examples/single_turn_summarizer.py) - Basic single-turn usage
- [`research_assistant.py`](examples/research_assistant.py) - Full agentic loop with tools

---

## 🧪 Testing

```bash
# Run all tests
pytest

# With coverage
pytest --cov=acorn

# Specific test file
pytest tests/test_agentic_loop.py -v
```

**Current status:** 128 tests passing, 90% coverage

---

## 📖 Documentation

- [Implementation Status](IMPLEMENTATION_STATUS.md) - Detailed feature status and API reference
- [Progress Summary](PROGRESS_SUMMARY.md) - Development progress and examples
- [Specifications](specs/) - Original design specifications

---

## 🛣️ Roadmap

### ✅ Completed (v0.1-beta)
- Single-turn execution
- Multi-turn agentic loops
- Tool calling with auto-schema generation
- Parse error recovery
- Dynamic tool management
- Step callbacks

### 🚧 In Progress
- Streaming responses (Phase 8)
- Forced termination strategies (Phase 7)

### 📋 Planned
- Branching workflows (Phase 9)
- Provider caching (Phase 10)
- Async support (v0.2)

---

## 🤝 Contributing

Contributions welcome! To continue implementation:

1. Follow phases 7-10 in order
2. Write tests first (maintain >80% coverage)
3. Update documentation
4. Add examples for new features

---

## 📄 License

MIT License - see [LICENSE](LICENSE) for details

---

## 🙏 Acknowledgments

- Heavily influenced by [DSPy](https://github.com/stanfordnlp/dspy)
- Built on [Pydantic](https://pydantic.dev/) and [LiteLLM](https://litellm.ai/)
- Developed with [Claude Code](https://claude.ai/code)

---

## 💬 Questions?

Check out:
- [Implementation Status](IMPLEMENTATION_STATUS.md) for detailed API docs
- [Examples](examples/) for working code
- [Tests](tests/) for usage patterns

---

**Status:** Production-ready for single-turn and multi-turn use cases
**Version:** 0.1.0-beta
**Last Updated:** 2026-01-24
