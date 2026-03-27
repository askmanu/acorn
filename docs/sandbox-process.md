---
title: ProcessSandbox
nav_order: 6
---

# ProcessSandbox

Code execution in a child process with timeout and kill support. Same whitelist model as `LocalSandbox` but runs in a subprocess via `multiprocessing`. Zero external dependencies.

> **Not safe for production use with untrusted code.** Same namespace restriction caveats as LocalSandbox — the process boundary adds crash isolation and timeouts but does not prevent sandbox escapes. For untrusted input, use [DockerSandbox](sandbox-docker.md) or a dedicated sandboxing service (E2B, Daytona).

## Quick Start

```python
from acorn import Module
from acorn.services import ProcessSandbox

class CodeAgent(Module):
    """Agent that runs code with timeout protection."""
    max_steps = 10
    tools = [ProcessSandbox(allowed_modules=["math"], timeout=10.0)]
    final_output = Output
```

The LLM gets `process_sandbox__execute` and `process_sandbox__reset` tools.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `allowed_modules` | `list[str]` | `None` | Module names the sandbox can import. |
| `allowed_builtins` | `list[str]` | `None` | Additional builtins beyond the safe default set. |
| `namespace` | `dict` | `None` | Objects to inject into the sandbox namespace. |
| `timeout` | `float` | `30.0` | Maximum execution time in seconds per `execute()` call. |

### Timeout Behavior

If code exceeds the timeout, the worker process is killed and automatically restarted:

```python
sandbox = ProcessSandbox(timeout=5.0)
await sandbox.setup()

# This times out after 5 seconds
result = sandbox.execute("while True: pass")
# Returns: {"success": false, "stderr": "Execution timed out after 5.0s. Sandbox has been reset."}

# Sandbox is usable again immediately
result = sandbox.execute("2 + 2")
# Returns: {"success": true, "return_value": "4"}
```

After a timeout, all state from the previous session is lost (the process was killed).

### Whitelist and Namespace

Works identically to `LocalSandbox`. See the [LocalSandbox docs](sandbox-local.md) for details on `allowed_modules`, `allowed_builtins`, and `namespace`.

```python
sandbox = ProcessSandbox(
    allowed_modules=["math", "json", "re"],
    namespace={"data": [1, 2, 3]},
    timeout=15.0,
)
```

## Persistent State

State persists across `execute()` calls as long as the process stays alive:

```python
sandbox.execute("total = 0")
sandbox.execute("total += 42")
sandbox.execute("total")  # return_value: "42"
```

State is lost when:
- `reset()` is called (clean restart)
- An execution times out (process is killed and restarted)

## Output Format

Same as `LocalSandbox` — a JSON string with `stdout`, `stderr`, `return_value`, and `success` fields.

## Pros and Cons

**Pros:**
- Zero dependencies — stdlib only (`multiprocessing`)
- Timeout support — hard kill on runaway code, automatic recovery
- Process isolation — separate memory space, crashes don't affect the host
- Persistent state — REPL-like experience within a session
- Same whitelist model as LocalSandbox — easy to switch between them

**Cons:**
- State lost on timeout — the process is killed, so all variables are gone. The sandbox restarts automatically but clean.
- Serialization constraints — objects in `namespace` must be picklable to cross the process boundary. Complex objects (database connections, file handles) won't work. Use `allowed_modules` and let the sandbox create them instead.
- Slightly higher overhead — process startup adds a small delay on first use and after timeouts.
- Namespace-only isolation — same restriction-bypass caveats as LocalSandbox. For untrusted code, use `DockerSandbox`.

## When to Use

ProcessSandbox is best for:
- Agent-generated code that might loop or hang (timeout protection)
- Scenarios where a crash shouldn't take down the host
- Same use cases as LocalSandbox but with an added safety net
- Long-running agent sessions where code quality varies
