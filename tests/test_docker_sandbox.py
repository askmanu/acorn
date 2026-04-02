"""Tests for the DockerSandbox service."""

import json
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest

from acorn.services.docker_sandbox import DockerSandbox


@pytest.fixture
def docker_sandbox():
    """Create a DockerSandbox (no setup — tests mock the runtime)."""
    return DockerSandbox(image="python:3.12-slim", timeout=30.0)


class TestDockerSandboxLifecycle:
    @pytest.mark.asyncio
    async def test_setup_checks_runtime(self):
        with patch("acorn.services.docker_sandbox.shutil.which", return_value="/usr/bin/docker"):
            sb = DockerSandbox()
            await sb.setup()  # should not raise

    @pytest.mark.asyncio
    async def test_setup_fails_without_runtime(self):
        with patch("acorn.services.docker_sandbox.shutil.which", return_value=None):
            sb = DockerSandbox()
            with pytest.raises(RuntimeError, match="not found in PATH"):
                await sb.setup()

    @pytest.mark.asyncio
    async def test_health_when_runtime_available(self):
        sb = DockerSandbox()
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert await sb.health() is True

    @pytest.mark.asyncio
    async def test_health_when_runtime_unavailable(self):
        sb = DockerSandbox()
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert await sb.health() is False


class TestDockerSandboxExecute:
    def test_simple_execution(self, docker_sandbox):
        mock_result = MagicMock()
        mock_result.stdout = "4\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            result = json.loads(docker_sandbox.execute("print(2 + 2)"))

        assert result["success"] is True
        assert result["stdout"] == "4\n"

        # Verify docker run was called with correct args
        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "python:3.12-slim" in cmd
        assert "print(2 + 2)" in cmd

    def test_error_captured(self, docker_sandbox):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = "NameError: name 'foo' is not defined\n"
        mock_result.returncode = 1

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result):
            result = json.loads(docker_sandbox.execute("foo"))

        assert result["success"] is False
        assert "NameError" in result["stderr"]

    def test_timeout_kills_container(self, docker_sandbox):
        """On timeout, the container should be force-killed."""
        mock_kill_result = MagicMock()

        def side_effect(*args, **kwargs):
            cmd = args[0]
            if cmd[1] == "run":
                raise subprocess.TimeoutExpired("docker", 30)
            # "kill" command
            return mock_kill_result

        with patch("acorn.services.docker_sandbox.subprocess.run", side_effect=side_effect) as mock_run:
            result = json.loads(docker_sandbox.execute("while True: pass"))

        assert result["success"] is False
        assert "timed out" in result["stderr"]

        # Verify docker kill was called
        calls = mock_run.call_args_list
        assert len(calls) == 2
        kill_cmd = calls[1][0][0]
        assert kill_cmd[1] == "kill"

    def test_network_disabled_by_default(self, docker_sandbox):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            docker_sandbox.execute("pass")

        cmd = mock_run.call_args[0][0]
        assert "--network" in cmd
        idx = cmd.index("--network")
        assert cmd[idx + 1] == "none"

    def test_network_enabled(self):
        sb = DockerSandbox(network=True)
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            sb.execute("pass")

        cmd = mock_run.call_args[0][0]
        assert "--network" not in cmd

    def test_memory_limit_set(self, docker_sandbox):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            docker_sandbox.execute("pass")

        cmd = mock_run.call_args[0][0]
        assert "--memory" in cmd
        idx = cmd.index("--memory")
        assert cmd[idx + 1] == "256m"

    def test_read_only_filesystem_by_default(self, docker_sandbox):
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            docker_sandbox.execute("pass")

        cmd = mock_run.call_args[0][0]
        assert "--read-only" in cmd
        assert "--tmpfs" in cmd
        idx = cmd.index("--tmpfs")
        assert cmd[idx + 1] == "/tmp"

    def test_read_only_disabled(self):
        sb = DockerSandbox(read_only=False)
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            sb.execute("pass")

        cmd = mock_run.call_args[0][0]
        assert "--read-only" not in cmd

    def test_workdir_mount(self):
        sb = DockerSandbox(workdir="/tmp/myproject")
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            sb.execute("pass")

        cmd = mock_run.call_args[0][0]
        assert "-v" in cmd
        idx = cmd.index("-v")
        assert cmd[idx + 1] == "/tmp/myproject:/workspace"
        assert "-w" in cmd

    def test_podman_runtime(self):
        sb = DockerSandbox(runtime="podman")
        mock_result = MagicMock()
        mock_result.stdout = "ok\n"
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            sb.execute("print('ok')")

        cmd = mock_run.call_args[0][0]
        assert cmd[0] == "podman"

    def test_container_name_is_unique(self, docker_sandbox):
        """Each execute() call should use a unique container name."""
        mock_result = MagicMock()
        mock_result.stdout = ""
        mock_result.stderr = ""
        mock_result.returncode = 0

        with patch("acorn.services.docker_sandbox.subprocess.run", return_value=mock_result) as mock_run:
            docker_sandbox.execute("pass")
            docker_sandbox.execute("pass")

        calls = mock_run.call_args_list
        name1 = calls[0][0][0][calls[0][0][0].index("--name") + 1]
        name2 = calls[1][0][0][calls[1][0][0].index("--name") + 1]
        assert name1 != name2
        assert name1.startswith("acorn-sandbox-")


class TestDockerSandboxReset:
    def test_reset_is_noop(self, docker_sandbox):
        result = docker_sandbox.reset()
        assert "stateless" in result


class TestDockerSandboxToolIntegration:
    def test_get_tools_returns_prefixed(self):
        sb = DockerSandbox()
        tools = sb.get_tools()
        names = [t.__name__ for t in tools]
        assert "docker_sandbox__execute" in names
        assert "docker_sandbox__reset" in names

    def test_tools_have_schemas(self):
        sb = DockerSandbox()
        tools = sb.get_tools()
        for t in tools:
            assert hasattr(t, "_tool_schema")
            schema = t._tool_schema
            func_schema = schema.get("function", schema)
            assert func_schema["name"].startswith("docker_sandbox__")
