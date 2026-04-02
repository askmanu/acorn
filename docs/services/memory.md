---
title: Memory
parent: Services
nav_order: 1
---

# Memory

SQLite-backed persistent storage for long-term memory. Save, search, and manage key-value entries with optional tags. Zero external dependencies.

## Quick Start

```python
from acorn import Module
from acorn.services.memory import Memory

class Agent(Module):
    """Agent with long-term memory."""
    max_steps = 10
    tools = [Memory(path="./agent_memory.db")]
    final_output = Output
```

The LLM gets four tools: `memory__save`, `memory__search`, `memory__delete`, and `memory__list_all`.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `path` | `str` | `"./memory.db"` | Path to the SQLite database file |

### Database Path

Store memories in a file:

```python
memory = Memory(path="./memories/agent.db")
```

Use an in-memory database for testing:

```python
memory = Memory(path=":memory:")
```

## Tools Provided

### memory__save

Save or update a memory entry by key. If the key exists, it's updated. If not, a new entry is created.

```python
# LLM calls:
memory__save(key="user_preference", content="User prefers dark mode", tags=["ui", "settings"])
```

**Parameters:**
- `key` (str): Unique identifier for the memory
- `content` (str): The content to store
- `tags` (list[str], optional): Tags for categorization

**Returns:** Confirmation message

### memory__search

Search memories by keyword. Matches against key, content, and tags.

```python
# LLM calls:
memory__search(query="dark mode", limit=5)
```

**Parameters:**
- `query` (str): Search query
- `limit` (int, default=5): Maximum number of results

**Returns:** JSON array of matching entries with `key`, `content`, and `tags` fields

### memory__delete

Delete a memory entry by key.

```python
# LLM calls:
memory__delete(key="user_preference")
```

**Parameters:**
- `key` (str): The key of the memory to delete

**Returns:** Confirmation message or "Memory not found" if key doesn't exist

### memory__list_all

List all stored memories, ordered by most recently updated.

```python
# LLM calls:
memory__list_all(limit=20)
```

**Parameters:**
- `limit` (int, default=20): Maximum number of entries to return

**Returns:** JSON array of all entries with `key`, `content`, and `tags` fields

## Memory Entry Format

Each memory entry has:

```json
{
  "key": "unique_identifier",
  "content": "The stored content",
  "tags": ["tag1", "tag2"]
}
```

- **key**: Unique identifier (string). Acts as primary key.
- **content**: Text content to store (string).
- **tags**: Optional list of strings for categorization (array, can be empty).

Entries also have internal timestamps (`created_at`, `updated_at`) used for sorting, but these are not exposed to the LLM.

## Common Patterns

### Pattern 1: User Preferences

Store and retrieve user preferences across sessions:

```python
from acorn.services.memory import Memory

class PersonalAssistant(Module):
    """Assistant that remembers user preferences."""
    max_steps = 10
    tools = [Memory(path="./user_prefs.db")]
    final_output = Response

# LLM stores preference:
# memory__save(key="timezone", content="America/New_York", tags=["settings"])

# Later, LLM searches:
# memory__search(query="timezone")
# Returns: [{"key": "timezone", "content": "America/New_York", "tags": ["settings"]}]
```

See `examples/personal_assistant.py` for a complete implementation.

### Pattern 2: Fact Database

Build a knowledge base the agent can query:

```python
memory = Memory(path="./facts.db")

class FactChecker(Module):
    """Agent that verifies facts against stored knowledge."""
    max_steps = 8
    tools = [memory, search_web]
    final_output = FactCheckResult

# LLM stores facts:
# memory__save(key="earth_radius", content="6,371 kilometers", tags=["science", "geography"])
# memory__save(key="python_created", content="1991 by Guido van Rossum", tags=["technology", "history"])

# Later, LLM checks a claim:
# memory__search(query="python created")
# Returns stored fact for verification
```

### Pattern 3: Session Memory

Track conversation context and decisions made during a session:

```python
memory = Memory(path="./session.db")

class ProjectManager(Module):
    """Agent that tracks project decisions."""
    max_steps = 15
    tools = [memory, list_tasks, update_status]
    final_output = ProjectUpdate

# LLM records decisions:
# memory__save(key="decision_2024_01_15_api", content="Decided to use REST API instead of GraphQL", tags=["decisions", "architecture"])

# LLM lists all decisions:
# memory__list_all(limit=50)
```

## When to Use

**Good use cases:**
- User preferences and settings that persist across sessions
- Building a knowledge base the agent can reference
- Recording decisions, facts, or context for future reference
- Storing structured data the agent needs to retrieve later

**Not recommended:**
- Temporary state within a single module execution (use module attributes instead)
- Large binary data (Memory stores text; use file storage for binary data)
- High-frequency updates (Memory uses SQLite; consider Redis for caching)

## Best Practices

**Use descriptive keys**: Make keys self-documenting for easier search and debugging.

```python
# Good
memory__save(key="user_timezone_preference", content="America/New_York")

# Less clear
memory__save(key="tz", content="America/New_York")
```

**Tag entries for categorization**: Tags enable filtered searches and logical grouping.

```python
memory__save(
    key="project_deadline",
    content="March 15, 2024",
    tags=["project", "deadline", "important"]
)

# Later search by tag
memory__search(query="deadline")
```

**Use search before save**: Check if a key exists before overwriting to avoid losing data.

```python
# LLM workflow:
# 1. memory__search(query="user_name")
# 2. If found, use existing; if not, memory__save(key="user_name", content="Alice")
```

**Limit search results**: Use the `limit` parameter to control how much context is returned.

```python
# Get top 3 most relevant results
memory__search(query="project", limit=3)
```

**Clean up stale data**: Use `memory__delete` to remove outdated entries.

```python
# Remove completed tasks
memory__delete(key="task_123_completed")
```

## Benefits

**Persistence**: Data survives across module executions and system restarts.

**Zero dependencies**: Pure SQLite, no external database required.

**Keyword search**: Built-in search across keys, content, and tags.

**Flexible schema**: Store any text content with optional tags for organization.

**Automatic timestamps**: Entries track creation and update times for sorting.

**ACID guarantees**: SQLite provides transaction safety and data integrity.
