---
title: Services & Tool Discovery
nav_order: 3
---

# Services & Tool Discovery

Services group related tools with shared configuration, authentication, and lifecycle management. Tool Discovery lets agents find tools on demand instead of loading all schemas into the prompt.

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

**Key conventions:**
- Class name becomes the service name (`Gmail`)
- Class docstring becomes the service description
- `@tool` decorated methods become tools
- `__init__` holds configuration (tokens, DB paths, etc.)

## Auto-Prefixing

When a Service is added to a Module's `tools` list, its tools are automatically prefixed with the snake_case service name to prevent conflicts:

| Class Name | Tool Method | Prefixed Name |
|---|---|---|
| `Gmail` | `send` | `gmail__send` |
| `GoogleCalendar` | `create_event` | `google_calendar__create_event` |
| `Memory` | `save` | `memory__save` |

This means two services can have methods with the same name without conflict:

```python
class Agent(Module):
    """Agent with multiple services."""
    tools = [Gmail(token="..."), Slack(token="...")]
    # gmail__send and slack__send coexist
```

### Cherry-Picking Tools

To use specific tools without the prefix, reference them directly from a service instance:

```python
gmail = Gmail(token="...")

class Agent(Module):
    tools = [gmail.send, gmail.search]
    # Tools are named "send" and "search" (no prefix)
```

### Mixing Approaches

Combine services, cherry-picked tools, and plain functions freely:

```python
gmail = Gmail(token="...")
memory = Memory(path="./mem.db")

class Agent(Module):
    tools = [
        search_web,          # Plain @tool function
        Gmail(token="..."),  # All Gmail tools (prefixed)
        memory.save,         # Single Memory tool (no prefix)
    ]
```

## Lifecycle

Services have async lifecycle hooks that are called automatically by the Module:

```python
class Database(Service):
    """Database connection pool."""

    def __init__(self, url: str):
        self.url = url
        self.pool = None

    async def setup(self):
        """Called when the module starts. Use for async initialization."""
        self.pool = await create_pool(self.url)

    async def teardown(self):
        """Called when the module finishes. Use for cleanup."""
        if self.pool:
            await self.pool.close()

    async def health(self) -> bool:
        """Check if the service is operational."""
        return self.pool is not None and not self.pool.is_closed
```

- **`setup()`** — Called before the first LLM call. Use for establishing connections, refreshing tokens, etc.
- **`teardown()`** — Called after the module finishes (even on error). Use for closing connections, flushing buffers.
- **`health()`** — Returns `True` if the service is operational. Override for custom health checks.

Services also work as async context managers:

```python
async with Database(url="postgresql://...") as db:
    # db.setup() called automatically
    tools = db.get_tools()
# db.teardown() called automatically
```

## Writing a Custom Service

Here's a complete custom service:

```python
from acorn import Service, tool

class Weather(Service):
    """Weather data from OpenWeatherMap API."""

    def __init__(self, api_key: str, units: str = "metric"):
        self.api_key = api_key
        self.units = units
        self._session = None

    async def setup(self):
        import aiohttp
        self._session = aiohttp.ClientSession()

    async def teardown(self):
        if self._session:
            await self._session.close()

    @tool
    def current(self, city: str) -> str:
        """Get current weather for a city.

        Args:
            city: City name (e.g., "London")
        """
        # Use self._session, self.api_key, self.units
        return f"Weather in {city}: 22°C, sunny"

    @tool
    def forecast(self, city: str, days: int = 3) -> str:
        """Get weather forecast for a city.

        Args:
            city: City name
            days: Number of days to forecast
        """
        return f"{days}-day forecast for {city}"
```

Use it in a module:

```python
class TravelAgent(Module):
    """Plan trips with weather awareness."""
    max_steps = 10
    tools = [Weather(api_key="...")]
    final_output = TripPlan
```

## Built-in Services

### Memory

SQLite-backed persistent storage. Located in `acorn.services.memory`.

```python
from acorn.services.memory import Memory

class Agent(Module):
    """Agent with long-term memory."""
    max_steps = 10
    tools = [Memory(path="./agent_memory.db")]
    final_output = Output
```

**Configuration:**
- `path` — Path to the SQLite database file. Use `":memory:"` for in-memory storage (useful for testing).

**Tools provided:**

| Tool | Description |
|---|---|
| `memory__save` | Save or update a memory entry by key, with optional tags |
| `memory__search` | Search memories by keyword (matches key, content, and tags) |
| `memory__delete` | Delete a memory entry by key |
| `memory__list_all` | List all stored memories |

Each entry has a `key` (unique identifier), `content` (text), and optional `tags` (list of strings) for categorization.

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
