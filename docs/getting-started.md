# Getting Started with Acorn

This guide walks you through installing Acorn and building your first two agents. You'll learn the basics of structured I/O and agentic loops.

**What you'll build:**
1. A sentiment classifier (single-turn)
2. A research assistant (multi-turn with tools)

**Time to complete:** 15-20 minutes

## Prerequisites

- Python 3.10 or higher
- Basic familiarity with Python and Pydantic
- API key for an LLM provider (Anthropic, OpenAI, or any LiteLLM-supported provider)

## Installation

Install Acorn from source:

```bash
git clone https://github.com/askmanu/acorn
cd acorn
pip install -e .
```

Set your API key as an environment variable:

```bash
# For Anthropic Claude (default)
export ANTHROPIC_API_KEY="your-key-here"

# Or for OpenAI
export OPENAI_API_KEY="your-key-here"

# Or any other LiteLLM-supported provider
```

Verify the installation:

```bash
python -c "import acorn; print(acorn.__version__)"
```

You should see the version number printed (e.g., `0.3.1`).

## Example 1: Sentiment Classifier

Build a simple sentiment classifier that takes text and returns structured sentiment analysis.

Create a file called `sentiment.py`:

```python
from pydantic import BaseModel, Field
from acorn import Module

class SentimentClassifier(Module):
    """Classify the sentiment of text as positive, negative, or neutral."""

    # Define input schema
    class Input(BaseModel):
        text: str = Field(description="Text to analyze")

    # Define output schema
    class Output(BaseModel):
        sentiment: str = Field(description="positive, negative, or neutral")
        confidence: float = Field(description="Confidence score between 0 and 1")
        explanation: str = Field(description="Brief explanation of the classification")

    initial_input = Input
    final_output = Output

    # Configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.3

# Use the classifier
classifier = SentimentClassifier()

result = classifier(text="I absolutely love this product! It exceeded all my expectations.")

print(f"Sentiment: {result.sentiment}")
print(f"Confidence: {result.confidence}")
print(f"Explanation: {result.explanation}")
```

Run it:

```bash
python sentiment.py
```

**Expected output:**
```
Sentiment: positive
Confidence: 0.95
Explanation: The text uses strong positive language like "absolutely love" and "exceeded all my expectations"
```

**What's happening:**
- The module makes a single LLM call (default behavior when `max_steps` is not set)
- Input is validated against the `Input` schema using Pydantic
- The LLM calls the internal `__finish__` tool with structured output
- Output is validated against the `Output` schema
- You get a typed Pydantic model instance back

## Example 2: Research Assistant

Build a research assistant that uses tools and iterates through multiple steps to gather information.

Create a file called `research.py`:

```python
from pydantic import BaseModel, Field
from acorn import Module, tool

class ResearchAssistant(Module):
    """Research assistant that gathers information using tools."""

    # Define input schema
    class Input(BaseModel):
        question: str = Field(description="Research question to investigate")

    # Define output schema
    class Output(BaseModel):
        answer: str = Field(description="Answer to the research question")
        key_findings: list[str] = Field(description="Key findings from research")
        sources_used: list[str] = Field(description="Tools and sources consulted")

    initial_input = Input
    final_output = Output

    # Enable agentic loop with max 5 steps
    max_steps = 5

    # Configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.3

    # Define tools the agent can use
    @tool
    def search_web(self, query: str) -> str:
        """Search the web for information.

        Args:
            query: The search query
        """
        # In a real implementation, call a search API
        # For this example, return mock data
        return f"Search results for '{query}': Python is a high-level programming language created by Guido van Rossum in 1991."

    @tool
    def get_statistics(self, topic: str) -> dict:
        """Get statistical information about a topic.

        Args:
            topic: The topic to get statistics for
        """
        # Mock statistics
        return {
            "topic": topic,
            "popularity_rank": 1,
            "users": "millions worldwide"
        }

    # Optional: Track progress with callback
    def on_step(self, step):
        """Called after each step in the loop."""
        print(f"\nStep {step.counter}:")
        print(f"  Tools called: {[tc.name for tc in step.tool_calls]}")

        # You can inspect or modify the step here
        return step

# Use the research assistant
assistant = ResearchAssistant()

result = assistant(question="When was Python created and why is it popular?")

print("\n=== Research Results ===")
print(f"Answer: {result.answer}")
print(f"\nKey Findings:")
for finding in result.key_findings:
    print(f"  - {finding}")
print(f"\nSources: {', '.join(result.sources_used)}")
```

Run it:

```bash
python research.py
```

**Expected output:**
```
Step 1:
  Tools called: ['search_web']

Step 2:
  Tools called: ['get_statistics']

Step 3:
  Tools called: ['__finish__']

=== Research Results ===
Answer: Python was created in 1991 by Guido van Rossum. It's popular due to its readability and extensive use in various fields.

Key Findings:
  - Python was created by Guido van Rossum in 1991
  - It ranks #1 in popularity among programming languages
  - Used by millions of developers worldwide

Sources: search_web, get_statistics
```

**What's happening:**
- The agent runs in a loop for up to 5 steps (`max_steps = 5`)
- At each step, the LLM decides which tool to call or whether to finish
- Tools are called automatically, and results are sent back to the LLM
- The `on_step` callback lets you track progress
- The agent finishes when it calls `__finish__` with structured output
- If it reaches step 5 without finishing, Acorn forces termination

## Key Concepts

### Module
The `Module` class is the foundation of Acorn. It encapsulates:
- Model configuration (which LLM to use, temperature, etc.)
- Input/output schemas (Pydantic models)
- Tools available to the agent
- Lifecycle hooks (`on_step`, `on_stream`)

### Single-Turn vs Multi-Turn
- **Single-turn** (`max_steps = None`, default): One LLM call, immediate response
- **Multi-turn** (`max_steps = N`): Agentic loop with up to N iterations

### Tools
Functions the LLM can call to gather information or take actions. Define tools with:
- The `@tool` decorator
- Type hints for parameters
- Docstrings for descriptions

Acorn automatically generates the tool schema from your function signature.

### Structured I/O
All inputs and outputs use Pydantic models:
- **Input validation**: Arguments are checked before the LLM runs
- **Output validation**: LLM responses are validated against your schema
- **Type safety**: You get typed Python objects, not raw strings

## Next Steps

Now that you've built your first agents, explore more advanced features:

- **[Core Concepts](../specs/core-concepts.md)** - Deep dive into modules, tools, and schemas
- **[Agentic Loop](../specs/agentic-loop.md)** - Learn about step callbacks, dynamic tools, and termination strategies
- **[Examples](../examples/)** - More example implementations
- **[API Reference](../specs/api-reference.md)** - Complete API documentation

### Common Next Tasks

**Add more tools to your agent:**
```python
@tool
def calculate(self, expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)
```

**Enable streaming for real-time output:**
```python
class StreamingAgent(Module):
    stream = True

    def on_stream(self, chunk):
        if chunk.content:
            print(chunk.content, end="", flush=True)
```

**Customize the system prompt:**
```python
class CustomAgent(Module):
    """You are a helpful assistant specialized in technical documentation.

    Guidelines:
    - Be concise and accurate
    - Provide code examples when relevant
    - Cite sources when possible
    """
```

**Control the loop with callbacks:**
```python
def on_step(self, step):
    # Add tools dynamically
    if step.counter == 2:
        step.add_tool(specialized_tool)

    # Early termination
    if enough_information_gathered:
        step.finish(answer="...", key_findings=[...], sources_used=[...])

    return step
```

## Troubleshooting

**"Module 'acorn' has no attribute 'Module'"**
- Make sure you installed Acorn correctly: `pip install -e .`
- Check your import: `from acorn import Module` (capital M)

**"API key not found"**
- Set your environment variable: `export ANTHROPIC_API_KEY="your-key-here"`
- Or pass it in the model config: `model = {"model": "anthropic/claude-sonnet-4-5-20250514", "api_key": "your-key"}`

**"ValidationError" when running the module**
- Check that your input matches the `Input` schema
- Verify that the LLM's output can be validated against your `Output` schema
- Increase `max_parse_retries` if the LLM needs more attempts to format correctly

**Agent uses all steps without finishing**
- Increase `max_steps` to give the agent more iterations
- Simplify your task or provide better tool descriptions
- Add an `on_step` callback to inspect what's happening

## Get Help

- **GitHub Issues**: [github.com/askmanu/acorn/issues](https://github.com/askmanu/acorn/issues)
- **Documentation**: [specs/](../specs/)
- **Examples**: [examples/](../examples/)
