---
title: Services
nav_order: 5
has_children: true
---

# Services

Services group related tools with shared configuration, authentication, and lifecycle management. Instead of defining standalone `@tool` functions that each manage their own credentials or connections, Services bundle them together with a common namespace.

## What is a Service?

A Service is a collection of related tools that share config and state:

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

**Key conventions:**
- Class name becomes the service name (`Weather`)
- Class docstring becomes the service description
- `@tool` decorated methods become tools
- `__init__` holds configuration (tokens, DB paths, etc.)

## Why Use Services?

**Shared configuration**: One API key or database connection for multiple related tools.

```python
# Without Services: Each tool manages its own connection
@tool
def send_email(to: str, subject: str):
    gmail = Gmail(token=GMAIL_TOKEN)  # Creates connection
    gmail.send(to, subject)
    gmail.close()

@tool
def search_email(query: str):
    gmail = Gmail(token=GMAIL_TOKEN)  # Creates connection again
    return gmail.search(query)
    gmail.close()

# With Services: One connection shared by all tools
class Gmail(Service):
    def __init__(self, token: str):
        self.token = token
        self._client = None
    
    async def setup(self):
        self._client = create_gmail_client(self.token)
    
    async def teardown(self):
        self._client.close()
    
    @tool
    def send(self, to: str, subject: str):
        return self._client.send(to, subject)
    
    @tool
    def search(self, query: str):
        return self._client.search(query)
```

**Lifecycle management**: Automatic setup and teardown for resources.

```python
class Database(Service):
    async def setup(self):
        """Called before first LLM call."""
        self.pool = await create_pool(self.url)
    
    async def teardown(self):
        """Called after module finishes (even on error)."""
        await self.pool.close()
```

**Namespace isolation**: Auto-prefixed tool names prevent conflicts.

```python
class Agent(Module):
    tools = [Gmail(token="..."), Slack(token="...")]
    # gmail__send and slack__send coexist without conflict
```

## Service vs Standalone Tools

| Aspect | Service | Standalone `@tool` |
|--------|---------|-------------------|
| **Shared state** | Yes — all tools access `self` | No — each call is isolated |
| **Lifecycle hooks** | `setup()` / `teardown()` called automatically | Manual management |
| **Tool naming** | Auto-prefixed (`gmail__send`) | Direct function name |
| **Configuration** | `__init__` parameters | Function parameters or globals |
| **Best for** | Related tools with shared resources | Independent utilities |

**When to use Services:**
- Multiple tools sharing API credentials, database connections, or file handles
- Tools requiring setup (connection pooling, authentication refresh)
- Tools requiring cleanup (closing connections, flushing buffers)
- Related functionality that belongs together conceptually

**When to use standalone tools:**
- Single independent function
- No shared state needed
- No setup/teardown required
- Simple utilities (calculations, string formatting)

## Available Services

Acorn provides these built-in services:

### Memory

SQLite-backed persistent storage for long-term memory. Save, search, and manage key-value entries with optional tags.

```python
from acorn.services.memory import Memory

class Agent(Module):
    max_steps = 10
    tools = [Memory(path="./agent_memory.db")]
```

**Tools:** `memory__save`, `memory__search`, `memory__delete`, `memory__list_all`

[Memory documentation →](memory.md)

### LocalSandbox

In-process Python code execution with whitelisted namespace. Zero dependencies, minimal overhead, persistent state.

```python
from acorn.services import LocalSandbox

class DataAgent(Module):
    max_steps = 10
    tools = [LocalSandbox(allowed_modules=["math", "json"])]
```

**Tools:** `local_sandbox__execute`, `local_sandbox__reset`

[LocalSandbox documentation →](sandbox-local.md)

### ProcessSandbox

Code execution in a child process with timeout and kill support. Same whitelist model as LocalSandbox but with process isolation.

```python
from acorn.services import ProcessSandbox

class CodeAgent(Module):
    max_steps = 10
    tools = [ProcessSandbox(allowed_modules=["math"], timeout=10.0)]
```

**Tools:** `process_sandbox__execute`, `process_sandbox__reset`

[ProcessSandbox documentation →](sandbox-process.md)

### DockerSandbox

Fully isolated code execution in Docker or Podman containers. Stateless execution with filesystem, network, and resource limits.

```python
from acorn.services import DockerSandbox

class CodeAgent(Module):
    max_steps = 10
    tools = [DockerSandbox(image="python:3.12-slim")]
```

**Tools:** `docker_sandbox__execute`, `docker_sandbox__reset`

[DockerSandbox documentation →](sandbox-docker.md)

## Using Services

### Auto-Prefixing

When a Service is added to a Module's `tools` list, its tools are automatically prefixed with the snake_case service name:

| Class Name | Tool Method | Prefixed Name |
|---|---|---|
| `Gmail` | `send` | `gmail__send` |
| `GoogleCalendar` | `create_event` | `google_calendar__create_event` |
| `Memory` | `save` | `memory__save` |

This prevents conflicts when multiple services have methods with the same name:

```python
class Agent(Module):
    tools = [Gmail(token="..."), Slack(token="...")]
    # gmail__send and slack__send coexist
```

### Cherry-Picking Tools

To use specific tools without the prefix, reference them directly:

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

Services have async lifecycle hooks called automatically by the Module:

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

**Hooks:**
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

To write your own service:

1. **Subclass `Service`**
2. **Add `@tool` methods** for the tools you want to expose
3. **Implement lifecycle hooks** (`setup()`, `teardown()`, `health()`) if needed
4. **Use `__init__` for configuration**

Example:

```python
from acorn import Service, tool

class GitHub(Service):
    """GitHub API integration."""

    def __init__(self, token: str, org: str = None):
        self.token = token
        self.org = org
        self._client = None

    async def setup(self):
        import aiohttp
        self._client = aiohttp.ClientSession(
            headers={"Authorization": f"Bearer {self.token}"}
        )

    async def teardown(self):
        if self._client:
            await self._client.close()

    @tool
    def list_repos(self, limit: int = 10) -> list[dict]:
        """List repositories for the organization.

        Args:
            limit: Maximum number of repositories to return
        """
        # Use self._client to make API calls
        return [{"name": "repo1"}, {"name": "repo2"}]

    @tool
    def create_issue(self, repo: str, title: str, body: str) -> dict:
        """Create a new issue in a repository.

        Args:
            repo: Repository name
            title: Issue title
            body: Issue body
        """
        return {"number": 123, "url": "https://github.com/..."}
```

Use it:

```python
class DevAssistant(Module):
    """Development assistant with GitHub integration."""
    max_steps = 10
    tools = [GitHub(token="ghp_...", org="my-org")]
    final_output = TaskReport
```

## Benefits

**Separation of concerns**: Configuration and tool logic are separate from the Module.

**Reusability**: Share services across multiple modules:

```python
gmail = Gmail(token="...")

class EmailAssistant(Module):
    tools = [gmail]

class ResearchAgent(Module):
    tools = [gmail, Memory(path="./research.db")]
```

**Testability**: Mock or stub services during testing:

```python
class MockGmail(Service):
    @tool
    def send(self, to: str, subject: str):
        return f"Mock: sent to {to}"

# Use MockGmail in tests
```

**Maintainability**: Update service implementation without touching module definitions.

**Type safety**: `__init__` parameters are typed and validated by Python.

## Next Steps

- [Memory service →](memory.md) — Persistent storage for long-term memory
- [LocalSandbox →](sandbox-local.md) — In-process code execution
- [ProcessSandbox →](sandbox-process.md) — Process-isolated code execution with timeouts
- [DockerSandbox →](sandbox-docker.md) — Container-isolated code execution
- [Tool Discovery →](../services.md#tool-discovery) — Search-based tool discovery for modules with many tools
