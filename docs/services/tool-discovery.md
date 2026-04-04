---
title: Tool Discovery
parent: Services
nav_order: 5
---

# Tool Discovery

When a module has many tools (10+), sending all tool schemas in every prompt wastes context and can confuse the LLM. Tool Discovery solves this by letting the LLM search for tools on demand.

## Enabling Tool Discovery

Set `tool_discovery = "search"` on your Module:

```python
class Agent(Module):
    """Agent with many tools."""
    tools = [Gmail(token="..."), Memory(path="./mem.db"), Slack(token="...")]
    tool_discovery = "search"
    max_steps = 15
    final_output = Output
```

## How It Works

With `tool_discovery = "search"`:

1. The LLM prompt includes only `search_tools` and `__finish__` tool schemas (not all tool schemas)
2. The LLM calls `search_tools(query="send email")` to find relevant tools
3. Matching tool schemas are returned to the LLM
4. The LLM can then call those tools normally
5. All tools remain callable at any time — only their schemas are deferred

## When to Use It

- **Many tools (10+):** Reduces prompt size and helps the LLM focus
- **Multiple services:** When combining several services with overlapping functionality
- **Dynamic tool sets:** When tools are added/removed during execution

## How Search Works

The `ToolRegistry` indexes tools by name and description using keyword matching:

- Tokenizes tool names and descriptions into searchable words
- Scores results by token overlap with the query
- Boosts matches on tool names over description-only matches
- Returns top results sorted by relevance

No external dependencies — pure Python keyword search.

## Example

```python
from acorn.services.memory import Memory

class KnowledgeWorker(Module):
    """Agent that manages knowledge across multiple services."""

    tools = [
        Memory(path="./knowledge.db"),
        Gmail(token="..."),
        Slack(token="..."),
        Jira(token="..."),
    ]
    tool_discovery = "search"
    max_steps = 20
    final_output = Report

# The LLM sees only search_tools and __finish__ initially.
# It searches for tools as needed:
#   search_tools(query="save note") -> returns memory__save schema
#   search_tools(query="send message slack") -> returns slack__send schema
```
