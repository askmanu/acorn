---
title: LocalSandbox
parent: Services
nav_order: 2
---

# LocalSandbox

In-process Python code execution with a whitelisted namespace. Uses `code.InteractiveConsole` from the standard library with restricted builtins. Zero external dependencies.

> **Not safe for production use with untrusted code.** Namespace restriction can be bypassed by a determined attacker (e.g. via `object.__subclasses__()`). Use this only for agent-generated code where you control the system prompt. For untrusted input, use [DockerSandbox](sandbox-docker.md) or a dedicated sandboxing service (E2B, Daytona).

## Quick Start

```python
from acorn import Module
from acorn.services import LocalSandbox

class DataAgent(Module):
    """Agent that can run Python code."""
    max_steps = 10
    tools = [LocalSandbox(allowed_modules=["math", "json"])]
    final_output = Output
```

The LLM gets `local_sandbox__execute` and `local_sandbox__reset` tools.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `allowed_modules` | `list[str]` | `None` | Module names the sandbox can import. `None` means no imports allowed. |
| `allowed_builtins` | `list[str]` | `None` | Additional builtins beyond the safe default set. |
| `namespace` | `dict` | `None` | Objects to inject into the sandbox namespace. |

### Allowing Imports

By default, no imports are allowed. Specify a whitelist:

```python
sandbox = LocalSandbox(allowed_modules=["math", "json", "re", "datetime"])
```

Only these modules can be imported. Attempting `import os` or any other module raises `ImportError`.

### Injecting Objects

Pre-inject authenticated clients, datasets, or any objects the LLM should have access to:

```python
from myapp.clients import gmail_client, db

sandbox = LocalSandbox(
    namespace={
        "gmail": gmail_client,
        "db": db,
        "config": {"max_retries": 3},
    }
)
```

The LLM's code can use `gmail`, `db`, and `config` as if they were already defined.

### Adding Builtins

The sandbox exposes a safe subset of builtins by default: `print`, `len`, `range`, `list`, `dict`, `set`, `tuple`, `str`, `int`, `float`, `bool`, `type`, `abs`, `min`, `max`, `sum`, `sorted`, `reversed`, `enumerate`, `zip`, `map`, `filter`, `isinstance`, `hasattr`, `getattr`, `round`, `repr`, `all`, `any`, `iter`, `next`.

To expose additional builtins:

```python
sandbox = LocalSandbox(allowed_builtins=["ord", "chr", "hex"])
```

Notably excluded by default: `open`, `exec`, `eval`, `compile`, `__import__`.

## Persistent State

State persists across `execute()` calls within a session:

```python
# LLM call 1
sandbox.execute("x = [1, 2, 3]")

# LLM call 2 — x is still available
sandbox.execute("x.append(4)")
sandbox.execute("len(x)")  # returns "4"
```

Use `reset()` to clear all state and restore the original namespace.

## Output Format

`execute()` returns a JSON string:

```json
{
  "stdout": "hello world\n",
  "stderr": "",
  "return_value": "'hello world'",
  "success": true
}
```

- **`stdout`** — Captured print output
- **`stderr`** — Error messages and tracebacks
- **`return_value`** — `repr()` of the expression result (or `"None"` for statements)
- **`success`** — `true` if execution completed without error

## Pros and Cons

**Pros:**
- Zero dependencies — stdlib only
- Minimal overhead — no process or container startup
- Persistent state across calls — true REPL experience
- Simplest setup — one line to configure
- Pre-injected namespace — give the LLM authenticated clients, data, etc.

**Cons:**
- No timeout support — a `while True` loop blocks forever. Use `ProcessSandbox` if you need timeouts.
- Namespace-only isolation — not a security sandbox. A determined attacker could escape via `object.__subclasses__()`. For untrusted code, use `DockerSandbox`.
- In-process — a segfault or crash in executed code takes down the host process.

## When to Use

LocalSandbox is best for:
- Agent-generated code where you control the system prompt
- Data analysis and computation tasks
- Quick prototyping and development
- Scenarios where startup overhead matters
