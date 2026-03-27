"""Docker/Podman sandbox service for fully isolated code execution.

.. warning::

    While DockerSandbox provides stronger isolation than LocalSandbox or
    ProcessSandbox, it still requires careful configuration for production
    use. Ensure images are trusted, network is disabled when not needed,
    and resource limits are appropriate. For adversarial inputs, consider
    gVisor-backed containers or a dedicated sandboxing service (E2B, Daytona).
"""

import asyncio
import json
import shutil
import subprocess
import uuid
from typing import Optional

from acorn.decorators import tool
from acorn.services.sandbox import Sandbox


class DockerSandbox(Sandbox):
    """Code execution in a Docker or Podman container.

    Shells out to the docker/podman CLI for full filesystem, network, and
    resource isolation. Each execute() call runs in a fresh container
    (stateless). No Python dependencies beyond the standard library.

    The container runs with a read-only root filesystem by default
    (``/tmp`` is mounted as a tmpfs for scratch space). Network access
    is disabled by default.

    .. warning::

        Not safe for production use with untrusted code without additional
        hardening. Use gVisor runtime, seccomp profiles, or a dedicated
        sandboxing service for adversarial inputs.
    """

    def __init__(
        self,
        image: str = "python:3.12-slim",
        timeout: float = 60.0,
        runtime: str = "docker",
        network: bool = False,
        memory_limit: str = "256m",
        workdir: Optional[str] = None,
        read_only: bool = True,
    ):
        """Initialize the DockerSandbox.

        Args:
            image: Docker image to use. Default: python:3.12-slim.
            timeout: Max execution time in seconds. Default: 60.
            runtime: Container runtime ("docker" or "podman"). Default: "docker".
            network: Whether to enable network access. Default: False.
            memory_limit: Container memory limit. Default: "256m".
            workdir: Host directory to bind-mount as /workspace. Default: None.
            read_only: Mount root filesystem as read-only. Default: True.
        """
        self._image = image
        self._timeout = timeout
        self._runtime = runtime
        self._network = network
        self._memory_limit = memory_limit
        self._workdir = workdir
        self._read_only = read_only

    async def setup(self):
        """Verify the container runtime is available."""
        if not shutil.which(self._runtime):
            raise RuntimeError(
                f"'{self._runtime}' not found in PATH. "
                f"Install {self._runtime} or use LocalSandbox/ProcessSandbox instead."
            )

    async def teardown(self):
        """No-op — containers are removed after each execution."""
        pass

    async def health(self) -> bool:
        """Check if the container runtime is available and responsive."""
        try:
            result = subprocess.run(
                [self._runtime, "info"],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False

    def _build_cmd(self, code: str, container_name: str) -> list[str]:
        """Build the docker run command."""
        cmd = [
            self._runtime, "run", "--rm",
            "--name", container_name,
            "--memory", self._memory_limit,
        ]

        if self._read_only:
            cmd.extend(["--read-only", "--tmpfs", "/tmp"])

        if not self._network:
            cmd.extend(["--network", "none"])

        if self._workdir:
            cmd.extend(["-v", f"{self._workdir}:/workspace", "-w", "/workspace"])

        cmd.extend([self._image, "python", "-c", code])
        return cmd

    def _kill_container(self, container_name: str):
        """Force-kill a container by name. Best-effort, ignores errors."""
        try:
            subprocess.run(
                [self._runtime, "kill", container_name],
                capture_output=True,
                timeout=10,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

    @tool
    def execute(self, code: str) -> str:
        """Execute Python code in a fresh container.

        Args:
            code: Python code to execute
        """
        container_name = f"acorn-sandbox-{uuid.uuid4().hex[:12]}"
        cmd = self._build_cmd(code, container_name)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            return json.dumps({
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_value": None,
                "success": result.returncode == 0,
            })
        except subprocess.TimeoutExpired:
            self._kill_container(container_name)
            return json.dumps({
                "stdout": "",
                "stderr": f"Execution timed out after {self._timeout}s.",
                "return_value": None,
                "success": False,
            })

    @tool
    def reset(self) -> str:
        """No-op — Docker sandbox is stateless."""
        return "Docker sandbox is stateless. Each execution starts fresh."
