---
title: Home
nav_order: 0
---

# Acorn

LLM agent framework with structured I/O. Built on Pydantic for schemas and LiteLLM for multi-provider access.

## What is Acorn?

Acorn makes it easy to build LLM agents that take typed inputs and return typed outputs. Define a `Module`, give it tools, and invoke it like a function.

```python
import asyncio
from acorn import Module
from pydantic import BaseModel, Field

class Output(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score 0-1")

class SentimentAnalyzer(Module):
    """Analyze sentiment of text."""
    final_output = Output

# Async
async def main():
    analyzer = SentimentAnalyzer()
    result = await analyzer(text="I love this product!")
    print(result.sentiment)  # "positive"

asyncio.run(main())
```

For synchronous code, use `.run()`:

```python
# Sync
analyzer = SentimentAnalyzer()
result = analyzer.run(text="I love this product!")
print(result.sentiment)  # "positive"
```

## Features

- **Structured I/O** - Pydantic models for inputs and outputs with automatic validation
- **Multi-provider** - Works with any LLM via LiteLLM (Anthropic, OpenAI, Google, Ollama, etc.)
- **Tool calling** - Define tools with `@tool` decorator, schemas auto-generated from type hints
- **Async tools** - Tools can be `async def` and execute concurrently when called together
- **Agentic loops** - Multi-step ReAct loops with configurable step limits
- **Branching** - Spawn sub-agents that inherit context and return structured results
- **Services** - Group related tools with shared config, auth, and lifecycle management
- **Tool Discovery** - Search-based tool discovery for agents with many tools
- **Streaming** - Stream text and partial structured outputs
- **Callbacks** - Inspect and modify execution at every step

## Quick links

- [Getting Started](getting-started) - Install and build your first agents
- [Module](module) - Full reference for the Module class
- [Branching](branching) - Sub-agents and parallel processing
- [Services](services/) - Tool collections, lifecycle, and built-in services
- [Tool Discovery](services) - Search-based tool discovery
- [GitHub](https://github.com/askmanu/acorn)
