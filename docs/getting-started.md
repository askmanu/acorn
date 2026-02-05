# Getting Started with acorn

Quick guide to install acorn and build your first agents.

## Prerequisites

- Python 3.10+
- API key for an LLM provider (Anthropic, OpenAI, or any LiteLLM-supported provider)

## Installation

```bash
pip install acorn
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

## Example 1: Single-Turn Agent

Create `sentiment.py`:

```python
from pydantic import BaseModel, Field
from acorn import Module

class Input(BaseModel):
    text: str = Field(description="Text to analyze")

class Output(BaseModel):
    sentiment: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(description="Confidence score between 0 and 1")

class SentimentClassifier(Module):
    """Classify the sentiment of text."""
    initial_input = Input
    final_output = Output
    model = "anthropic/claude-sonnet-4-5-20250514"

classifier = SentimentClassifier()
result = classifier(text="I love this product!")

print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")
```

Run it:

```bash
python sentiment.py
```

## Example 2: Multi-Turn Agent with Tools

Create `research.py`:

```python
from pydantic import BaseModel, Field
from acorn import Module, tool

class Input(BaseModel):
    question: str = Field(description="Research question")

class Output(BaseModel):
    answer: str = Field(description="Answer to the question")
    sources: list[str] = Field(description="Sources consulted")

class ResearchAssistant(Module):
    """Research assistant with tools."""
    initial_input = Input
    final_output = Output
    max_steps = 5
    model = "anthropic/claude-sonnet-4-5-20250514"

    @tool
    def get_info(self, topic: str) -> str:
        """Get information about a topic."""
        return f"Information about {topic}"

    @tool
    def get_stats(self, topic: str) -> dict:
        """Get statistics about a topic."""
        return {"topic": topic, "rank": 1}

assistant = ResearchAssistant()
result = assistant(question="Tell me about Python")

print(f"Answer: {result.answer}")
print(f"Sources: {result.sources}")
```

Run it:

```bash
python research.py
```

## What's Next

- Define tools with `@tool` decorator
- Use `max_steps` to enable multi-turn loops
- All inputs/outputs use Pydantic for validation