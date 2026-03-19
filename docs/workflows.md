---
title: Workflows
nav_order: 4
---

# Workflows

Acorn modules compose naturally using standard Python. You don't need a graph DSL or workflow engine to build multi-step pipelines, conditional routing, or parallel execution. Python's control flow is the workflow.

## Why no graph API?

Graph-based workflow frameworks (like LangGraph) introduce nodes, edges, and state machines as first-class concepts. Acorn takes a different approach: **modules are the nodes, Python is the graph.**

This is a deliberate choice:

- **Modules already encapsulate LLM calls** with typed I/O, tools, and lifecycle hooks
- **Python already has** sequencing (`await`), branching (`if/else`), parallelism (`asyncio.gather`), and loops (`for`/`while`)
- **LLM agents are already routers** — an agent with tools decides its own next step at each iteration, which is more flexible than a static graph

Adding a graph abstraction on top would duplicate what Python and the agentic loop already provide.

## Patterns

### Sequential pipeline

Pass output from one module into the next:

```python
from pydantic import BaseModel, Field
from acorn import Module

class RawText(BaseModel):
    text: str

class ExtractedData(BaseModel):
    entities: list[str]
    summary: str

class QualityReport(BaseModel):
    score: float
    issues: list[str]

class Extractor(Module):
    """Extract entities and summarize text."""
    initial_input = RawText
    final_output = ExtractedData

class QualityChecker(Module):
    """Evaluate extraction quality."""
    initial_input = ExtractedData
    final_output = QualityReport

async def pipeline(text: str):
    extracted = await Extractor()(text=text)
    report = await QualityChecker()(entities=extracted.entities, summary=extracted.summary)
    return report
```

Each module validates its own inputs and outputs. If `Extractor` returns malformed data, `QualityChecker` catches it at the Pydantic boundary.

### Conditional routing

Use Python control flow to route between modules based on results:

```python
class TriageOutput(BaseModel):
    category: str = Field(description="billing, technical, or general")
    urgency: str = Field(description="low, medium, or high")

class Triage(Module):
    """Classify incoming support tickets."""
    initial_input = TicketInput
    final_output = TriageOutput

class BillingAgent(Module):
    """Handle billing inquiries."""
    initial_input = TicketInput
    final_output = Resolution
    max_steps = 5

class TechnicalAgent(Module):
    """Handle technical issues."""
    initial_input = TicketInput
    final_output = Resolution
    max_steps = 10
    tools = [search_docs, check_status]

class GeneralAgent(Module):
    """Handle general inquiries."""
    initial_input = TicketInput
    final_output = Resolution

async def handle_ticket(ticket: str):
    triage = await Triage()(ticket=ticket)

    routes = {
        "billing": BillingAgent,
        "technical": TechnicalAgent,
        "general": GeneralAgent,
    }

    agent_class = routes[triage.category]
    return await agent_class()(ticket=ticket)
```

This is equivalent to a conditional edge in a graph framework, but easier to read, debug, and extend.

### Parallel execution

Run independent modules concurrently with `asyncio.gather`:

```python
import asyncio

class SentimentAnalyzer(Module):
    """Analyze sentiment."""
    initial_input = TextInput
    final_output = SentimentOutput

class TopicClassifier(Module):
    """Classify topics."""
    initial_input = TextInput
    final_output = TopicOutput

class ToxicityChecker(Module):
    """Check for toxic content."""
    initial_input = TextInput
    final_output = ToxicityOutput

async def analyze(text: str):
    sentiment, topics, toxicity = await asyncio.gather(
        SentimentAnalyzer()(text=text),
        TopicClassifier()(text=text),
        ToxicityChecker()(text=text),
    )
    return {
        "sentiment": sentiment,
        "topics": topics,
        "toxicity": toxicity,
    }
```

All three modules run simultaneously. Each gets its own LLM call and validates independently.

### Fan-out / fan-in (map-reduce)

Process a list of items in parallel and aggregate results:

```python
import asyncio
from pydantic import BaseModel, Field

class ArticleInput(BaseModel):
    url: str

class ArticleSummary(BaseModel):
    url: str
    title: str
    key_points: list[str]

class Summarizer(Module):
    """Summarize a single article."""
    initial_input = ArticleInput
    final_output = ArticleSummary

class DigestOutput(BaseModel):
    overview: str
    articles: list[str]

class DigestWriter(Module):
    """Write a digest from multiple article summaries."""
    final_output = DigestOutput

async def daily_digest(urls: list[str]):
    # Fan-out: summarize all articles in parallel
    summaries = await asyncio.gather(
        *[Summarizer()(url=url) for url in urls]
    )

    # Fan-in: combine summaries into a digest
    context = "\n\n".join(
        f"## {s.title}\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in summaries
    )

    return await DigestWriter()(summaries=context)
```

For map-reduce within an agentic loop, use [branches](branching.md) instead. The agent calls `branch()` for each item and aggregates results across steps.

### Loops and retries

Use Python loops for iterative refinement:

```python
class Draft(BaseModel):
    content: str

class Review(BaseModel):
    approved: bool
    feedback: str

class Writer(Module):
    """Write content based on a brief and optional feedback."""
    final_output = Draft

class Reviewer(Module):
    """Review content for quality."""
    final_output = Review

async def write_with_review(brief: str, max_revisions: int = 3):
    feedback = None

    for i in range(max_revisions):
        draft = await Writer()(brief=brief, feedback=feedback)
        review = await Reviewer()(content=draft.content, brief=brief)

        if review.approved:
            return draft

        feedback = review.feedback

    return draft  # Return best effort after max revisions
```

### LLM-driven routing (agentic)

For cases where the routing logic itself requires reasoning, let the agent decide. This is what the agentic loop is built for:

```python
class ResearchAgent(Module):
    """Research a topic using available tools.

    Choose the best tool for each step. When you have
    enough information, compile your findings.
    """
    max_steps = 10
    tools = [search_web, search_academic, query_database, calculate]
    final_output = ResearchOutput

    def on_step(self, step):
        print(f"Step {step.counter}: {[tc.name for tc in step.tool_calls]}")
        return step
```

The agent examines results at each step and decides what to do next. No edges or routing rules needed — the LLM *is* the router.

### Mixed: deterministic pipeline with agentic steps

Combine fixed pipeline structure with agentic modules at individual steps:

```python
class Researcher(Module):
    """Gather information on a topic."""
    max_steps = 10
    tools = [search_web, search_academic]
    final_output = ResearchFindings

class Synthesizer(Module):
    """Synthesize findings into a coherent report."""
    final_output = Report

class FactChecker(Module):
    """Verify claims in the report."""
    max_steps = 5
    tools = [verify_claim, check_source]
    final_output = VerifiedReport

async def research_pipeline(topic: str):
    # Step 1: Agentic research (LLM decides which tools to use)
    findings = await Researcher()(topic=topic)

    # Step 2: Single-turn synthesis (deterministic)
    report = await Synthesizer()(findings=findings.data, topic=topic)

    # Step 3: Agentic fact-checking (LLM decides what to verify)
    verified = await FactChecker()(report=report.content)

    return verified
```

The pipeline itself is deterministic — always research, then synthesize, then verify. But individual steps are agentic, using tools and multiple LLM calls internally.

## When to use what

| Pattern | Use when | Example |
|---------|----------|---------|
| Sequential pipeline | Steps have a fixed order | ETL, content pipelines |
| Conditional routing | Next step depends on a classification | Support ticket routing |
| Parallel execution | Independent analyses of the same input | Multi-aspect evaluation |
| Fan-out / fan-in | Same operation on many items | Batch summarization |
| Loops | Iterative refinement until quality bar met | Write-review cycles |
| LLM-driven routing | Routing requires reasoning | Research, open-ended tasks |
| Mixed | Fixed structure, flexible steps | Research pipelines |

## Comparison with graph frameworks

| Aspect | Graph frameworks | Acorn + Python |
|--------|-----------------|----------------|
| **Routing** | Explicit edges, conditional edges | `if/else`, or let the LLM decide |
| **State** | Shared state object passed between nodes | Module I/O (typed, validated) |
| **Parallelism** | Parallel nodes in graph | `asyncio.gather` |
| **Cycles** | Cycle edges in graph | `for`/`while` loops, or `max_steps` |
| **Observability** | Framework-level tracing | `on_step` callbacks, print statements |
| **Debugging** | Graph visualization | Standard Python debugging |
| **Type safety** | Varies | Pydantic validation at every boundary |

Graph frameworks add value when you need visual workflow editors, built-in persistence and replay, or integration with a specific orchestration platform. If your workflows are defined in code and run in Python, the patterns above are simpler and more flexible.

## Best practices

**Keep modules focused.** Each module should do one thing. A pipeline of five focused modules is easier to test and debug than one module trying to do everything.

**Validate at boundaries.** Define `initial_input` and `final_output` on every module. This turns each connection in your pipeline into a type-checked contract.

**Use async throughout.** All the patterns above use `await` and `asyncio.gather`. This gives you free parallelism and keeps the code consistent.

**Start simple.** A sequential pipeline with two modules is a valid workflow. Add parallelism, routing, and loops only when you need them.

**Reuse module instances.** Module instances are lightweight. Create them once and call them multiple times:

```python
summarizer = Summarizer()

results = await asyncio.gather(
    *[summarizer(text=text) for text in texts]
)
```

**Handle errors at the pipeline level.** Each module raises typed exceptions (`ParseError`, `AcornError`). Catch and handle them in your pipeline function, not inside modules:

```python
async def pipeline(text: str):
    try:
        extracted = await Extractor()(text=text)
    except ParseError:
        # Fallback or retry with different model
        extracted = await Extractor(model="openai/gpt-4o")(text=text)

    return await Processor()(data=extracted.data)
```

## Next steps

- See [Module](module.md) for details on module configuration
- See [Branching](branching.md) for LLM-driven sub-agent delegation
- Check [Getting Started](getting-started.md) for basic examples
