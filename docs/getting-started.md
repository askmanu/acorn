# Getting Started with acorn

## Installation

Install acorn with pip:

```bash
pip install acorn
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="your-key-here"
# or
export OPENAI_API_KEY="your-key-here"
```

Requires Python 3.10+.

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
    initial_input = Input
    final_output = Output
    model = "anthropic/claude-sonnet-4-5-20250514"

result = SentimentClassifier()(text="I love this product!")
print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")
```

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

result = ResearchAssistant()(question="Tell me about Python")
print(f"Answer: {result.answer}")
print(f"Sources: {result.sources}")
```

```bash
python research.py
```