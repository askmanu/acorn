---
title: DockerSandbox
nav_order: 7
---

# DockerSandbox

Fully isolated code execution in Docker or Podman containers. Each `execute()` call runs in a fresh container with filesystem, network, and resource limits. No Python dependencies — shells out to the container runtime CLI.

> **Use with caution in production.** DockerSandbox provides stronger isolation than LocalSandbox/ProcessSandbox but is not hardened against all container escape vectors. For adversarial inputs, use gVisor runtime (`--runtime=runsc`), seccomp profiles, or a dedicated sandboxing service (E2B, Daytona).

## Quick Start

```python
from acorn import Module
from acorn.services import DockerSandbox

class CodeAgent(Module):
    """Agent that runs code in isolated containers."""
    max_steps = 10
    tools = [DockerSandbox(image="python:3.12-slim")]
    final_output = Output
```

The LLM gets `docker_sandbox__execute` and `docker_sandbox__reset` tools.

**Prerequisite:** Docker or Podman must be installed and accessible in `PATH`.

## Configuration

| Parameter | Type | Default | Description |
|---|---|---|---|
| `image` | `str` | `"python:3.12-slim"` | Docker image to use. |
| `timeout` | `float` | `60.0` | Maximum execution time in seconds. |
| `runtime` | `str` | `"docker"` | Container runtime (`"docker"` or `"podman"`). |
| `network` | `bool` | `False` | Whether to enable network access. |
| `memory_limit` | `str` | `"256m"` | Container memory limit. |
| `workdir` | `str` | `None` | Host directory to bind-mount as `/workspace`. |
| `read_only` | `bool` | `True` | Mount root filesystem as read-only (`/tmp` available as tmpfs). |

### Using Podman

```python
sandbox = DockerSandbox(runtime="podman", image="python:3.12-slim")
```

Podman is rootless by default and accepts the same CLI flags as Docker.

### Enabling Network Access

Network is disabled by default (`--network=none`):

```python
# Allow the container to make HTTP requests
sandbox = DockerSandbox(network=True)
```

### Custom Images

Use any Docker image with Python installed:

```python
# Image with data science libraries pre-installed
sandbox = DockerSandbox(image="my-registry/python-data:latest")

# Image with specific SDK
sandbox = DockerSandbox(image="my-registry/python-gmail:latest")
```

### Mounting a Working Directory

Bind-mount a host directory into the container as `/workspace`:

```python
sandbox = DockerSandbox(workdir="/path/to/project")
```

The LLM's code can read/write files in `/workspace`. The container's working directory is set to `/workspace`.

### Resource Limits

```python
sandbox = DockerSandbox(
    memory_limit="512m",  # 512MB memory limit
    timeout=120.0,        # 2 minute execution timeout
)
```

### Security Defaults

The container runs with these security defaults:
- **Read-only root filesystem** — prevents writing to system directories. `/tmp` is available as a tmpfs for scratch space. Disable with `read_only=False` if needed.
- **No network** — `--network=none` by default. Enable with `network=True`.
- **Memory limit** — 256MB by default.
- **Container killed on timeout** — if execution exceeds the timeout, the container is force-killed (not just the subprocess).
- **Auto-removed** — `--rm` flag ensures no container artifacts are left behind.

## Stateless Execution

Each `execute()` call runs in a fresh `docker run --rm` container. There is no state persistence between calls:

```python
sandbox.execute("x = 42")
sandbox.execute("print(x)")  # NameError — x doesn't exist in this container
```

The `reset()` tool is a no-op since each execution already starts clean.

To share data between calls, use a mounted `workdir` and write/read files.

## Output Format

Same as other sandbox services — a JSON string with `stdout`, `stderr`, `return_value`, and `success` fields. Note that `return_value` is always `null` for DockerSandbox since output is captured from the container's stdout/stderr streams.

```json
{
  "stdout": "42\n",
  "stderr": "",
  "return_value": null,
  "success": true
}
```

## Pros and Cons

**Pros:**
- Full container isolation — filesystem, network, memory, and process isolation
- Resource limits — control memory, network access, and execution time
- Any image — use pre-built images with any libraries or SDKs installed
- No namespace bypass — unlike Local/ProcessSandbox, there are no Python-level escape routes
- Rootless option — Podman runs without root privileges

**Cons:**
- Requires Docker or Podman installed — not a pure Python solution
- High overhead per call — container startup adds 0.5-2s per execution
- Stateless — no persistent variables between calls. Use file-based state via `workdir` if needed.
- No whitelist model — isolation comes from the container, not namespace restriction. The code inside has full access to whatever the image provides.
- No return value capture — only stdout/stderr are captured. Use `print()` for output.

## When to Use

DockerSandbox is best for:
- Untrusted or adversarial code where namespace restriction isn't enough
- Running code that needs specific system-level dependencies or libraries
- Scenarios requiring strict resource limits (memory, network, CPU)
- Production deployments where isolation guarantees matter
- When you'd otherwise use E2B or Daytona but want a self-hosted option
