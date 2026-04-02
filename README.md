
<img width="851" height="190" alt="github" src="https://github.com/user-attachments/assets/3f06caa7-b670-4cfb-8a57-0278f7f946a0" />


**LLM agent framework with structured I/O**

Build AI agents with type-safe inputs and outputs, automatic tool calling, and powerful agentic loops.

[![Tests](https://img.shields.io/badge/tests-201%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-85%25-brightgreen)](tests/)
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
- 🌊 **Streaming Responses** - Real-time output with partial structured updates
- 💾 **Provider Caching** - Reduce latency and cost with prompt caching
- 🛡️ **Model Fallbacks** - Automatic provider failover for high availability
- 🌳 **Branching Workflows** - Spawn sub-agents that extend parent capabilities for parallel analysis and map-reduce patterns

---

> [!WARNING]
> Breaking Change in v0.8.0: Async-First API.
> 
> Acorn v0.8.0 makes `__call__` async. If you're upgrading from v0.7.x, you need to update your code:

- **Sync code**: Replace `agent(...)` with `agent.run(...)`
- **Async code**: Add `await` — `await agent(...)`

**Auto-migrate with your AI coding agent:**

Paste this into your AI coding agent to update your codebase automatically:

```
Find all usages of acorn Module subclasses being called directly (e.g. `agent(...)`, `MyModule()(...)`).
- If the call site is in a sync function or top-level code, replace it with `agent.run(...)`.
- If the call site is in an async function, add `await` before the call: `await agent(...)`.
Do NOT change the class definitions themselves, only the call sites.
```
---

## 🚀 Quick Start

### Installation

```bash
pip install acorn
```

Set your API key:

```bash
# For Anthropic Claude
export ANTHROPIC_API_KEY="your-key-here"

# Or for OpenAI
export OPENAI_API_KEY="your-key-here"

# Or any other LiteLLM-supported provider
```

---

### Single-Turn Example

```python
import asyncio
from pydantic import BaseModel, Field
from acorn import Module

class Input(BaseModel):
    text: str = Field(description="The text to summarize")
    max_words: int = Field(default=100, description="Maximum words in summary")

class Output(BaseModel):
    summary: str = Field(description="The concise summary")
    word_count: int = Field(description="Number of words in summary")

class Summarizer(Module):
    """Summarize text concisely."""

    initial_input = Input
    final_output = Output
    model = "anthropic/claude-sonnet-4-5-20250514"

# Use it (async)
async def main():
    summarizer = Summarizer()
    result = await summarizer(
        text="Long article text here...",
        max_words=50
    )
    print(result.summary)
    print(f"Words: {result.word_count}")

asyncio.run(main())
```

For synchronous code, use `.run()`:

```python
# Use it (sync)
summarizer = Summarizer()
result = summarizer.run(
    text="Long article text here...",
    max_words=50
)
print(result.summary)
print(f"Words: {result.word_count}")
```

### Multi-Turn Agentic Loop

```python
import asyncio
from pydantic import BaseModel, Field
from acorn import Module, tool

class Input(BaseModel):
    topic: str = Field(description="Research topic")
    depth: str = Field(default="shallow", description="Research depth")

class Output(BaseModel):
    findings: str = Field(description="Summary of findings")
    sources: list[str] = Field(description="Sources consulted")

class ResearchAgent(Module):
    """Research assistant with tools."""

    initial_input = Input
    max_steps = 5  # Enable agentic loop
    final_output = Output
    model = "anthropic/claude-sonnet-4-5-20250514"

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

# Use it (async)
async def main():
    agent = ResearchAgent()
    result = await agent(topic="Large Language Models", depth="shallow")

asyncio.run(main())
```

For synchronous code:

```python
# Use it (sync)
agent = ResearchAgent()
result = agent.run(topic="Large Language Models", depth="shallow")
```

---

## 📖 Documentation

**[askmanu.github.io/acorn](https://askmanu.github.io/acorn)**

- [Getting Started](https://askmanu.github.io/acorn/getting-started) - Installation and first steps
- [Module Reference](https://askmanu.github.io/acorn/module) - Complete Module API documentation
- [Branching](https://askmanu.github.io/acorn/branching) - Sub-agents and parallel processing

---

## 📚 Core Concepts

### Module
Base class for LLM agents. Configure with:
- `model` - LLM to use (required - no default)
- `temperature` - Sampling temperature
- `max_tokens` - Maximum tokens to generate
- `max_steps` - Max agentic loop iterations (None = single-turn)
- `initial_input` - Pydantic model for input schema
- `final_output` - Pydantic model for output schema
- `tools` - List of available tools
- `cache` - Enable provider-level prompt caching
- `model_fallbacks` - List of fallback models for automatic failover

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

Tools can be async:

```python
@tool
async def fetch_data(url: str) -> dict:
    """Fetch data from a URL.

    Args:
        url: The URL to fetch from
    """
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

When the LLM calls multiple tools in one step, they execute concurrently via `asyncio.gather()` for better performance.

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

Try them live on the [Gradio app](https://askmanu-acorn.hf.space) or browse the source in `examples/`:

| Example | Category | Description |
|---------|----------|-------------|
| [Simple Q&A](https://askmanu-acorn.hf.space/simple_qa) | Basic | Single-turn question answering with structured output |
| [HN Production Readiness](https://askmanu-acorn.hf.space/hn_production_check) | Agentic | Checks if a trending HN project is production-ready |
| [Documentation Coverage](https://askmanu-acorn.hf.space/doc_coverage) | Agentic | Scores documentation coverage of a GitHub repo (0–100) |
| [Bus Factor Calculator](https://askmanu-acorn.hf.space/bus_factor) | Branching | Calculates the bus factor of a GitHub repository |
| [License Compatibility](https://askmanu-acorn.hf.space/license_checker) | Agentic | Checks dependency license compatibility for conflicts |
| [Dependency Bloat Scanner](https://askmanu-acorn.hf.space/dependency_scanner) | Branching | Finds redundant and overlapping libraries in your deps |

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

**Current status:** 201 tests passing, 85% coverage

---

## 🛣️ Roadmap

### ✅ Completed
- Single-turn execution
- Multi-turn agentic loops
- Tool calling with auto-schema generation
- Parse error recovery
- Dynamic tool management
- Step callbacks
- Streaming responses with partial structured output
- Forced termination strategies
- Provider caching
- Model fallbacks
- Branching workflows
- Async-first API with sync wrapper

### 📋 Planned
- More docs
- Integration examples with different providers (vector DBs, observability tools, etc.)

---

## 🤝 Contributing

Contributions welcome! Please:

1. Check open issues for areas to help
2. Write tests for new features (maintain >80% coverage)
3. Update documentation
4. Add examples for new features

---

## 🙏 Acknowledgments
Thanks to @rosenbrockc for donating the `acorn` pip package name.

---

## 📄 License

[MIT License](LICENSE)
