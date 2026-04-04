---
title: Services & Tool Discovery
nav_order: 4
---

# Services & Tool Discovery

Services group related tools with shared configuration, authentication, and lifecycle management. Tool Discovery lets agents find tools on demand instead of loading all schemas into the prompt.

For detailed information about services, see the [Services category](services/).

## What is a Service?

A Service is a collection of related tools that share config and state. Instead of defining standalone `@tool` functions that each manage their own credentials or connections, a Service bundles them together.

```python
from acorn import Service, tool

class Gmail(Service):
    """Google Gmail integration."""

    def __init__(self, token: str):
        self.token = token

    @tool
    def send(self, to: str, subject: str, body: str) -> str:
        """Send an email.

        Args:
            to: Recipient email address
            subject: Email subject line
            body: Email body text
        """
        # Use self.token for authentication
        return f"Sent to {to}"

    @tool
    def search(self, query: str, limit: int = 10) -> str:
        """Search emails by query.

        Args:
            query: Search query
            limit: Maximum results to return
        """
        return f"Found results for: {query}"
```

See the [Services documentation](services/) for more details on writing custom services, lifecycle management, and available built-in services.

## Built-in Services

Acorn provides four built-in services:

- **[Memory](services/memory.md)** — SQLite-backed persistent storage for long-term memory
- **[LocalSandbox](services/sandbox-local.md)** — In-process Python code execution with whitelisted namespace
- **[ProcessSandbox](services/sandbox-process.md)** — Code execution in a child process with timeout support
- **[DockerSandbox](services/sandbox-docker.md)** — Fully isolated code execution in Docker/Podman containers

## Tool Discovery

When a module has many tools (10+), sending all tool schemas in every prompt wastes context and can confuse the LLM. Tool Discovery solves this by letting the LLM search for tools on demand.

### Enabling Tool Discovery

Set `tool_discovery = "search"` on your Module:

```python
class Agent(Module):
    """Agent with many tools."""
    tools = [Gmail(token="..."), Memory(path="./mem.db"), Slack(token="...")]
    tool_discovery = "search"
    max_steps = 15
    final_output = Output
```

### How It Works

With `tool_discovery = "search"`:

1. The LLM prompt includes only `search_tools` and `__finish__` tool schemas (not all tool schemas)
2. The LLM calls `search_tools(query="send email")` to find relevant tools
3. Matching tool schemas are returned to the LLM
4. The LLM can then call those tools normally
5. All tools remain callable at any time — only their schemas are deferred

### When to Use It

- **Many tools (10+):** Reduces prompt size and helps the LLM focus
- **Multiple services:** When combining several services with overlapping functionality
- **Dynamic tool sets:** When tools are added/removed during execution

### How Search Works

The `ToolRegistry` indexes tools by name and description using keyword matching:

- Tokenizes tool names and descriptions into searchable words
- Scores results by token overlap with the query
- Boosts matches on tool names over description-only matches
- Returns top results sorted by relevance

No external dependencies — pure Python keyword search.

### Example

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
