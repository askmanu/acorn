"""Tests for LocalSandbox and ProcessSandbox services."""

import json
import pytest

from acorn.services.sandbox import LocalSandbox, ProcessSandbox


# ---------------------------------------------------------------------------
# LocalSandbox
# ---------------------------------------------------------------------------


@pytest.fixture
async def sandbox():
    """Create a basic LocalSandbox with no extra modules."""
    sb = LocalSandbox()
    await sb.setup()
    yield sb
    await sb.teardown()


@pytest.fixture
async def sandbox_with_modules():
    """Create a LocalSandbox with math module allowed."""
    sb = LocalSandbox(allowed_modules=["math", "json"])
    await sb.setup()
    yield sb
    await sb.teardown()


@pytest.fixture
async def sandbox_with_namespace():
    """Create a LocalSandbox with injected namespace."""
    sb = LocalSandbox(namespace={"data": [1, 2, 3], "greeting": "hello"})
    await sb.setup()
    yield sb
    await sb.teardown()


class TestLocalSandboxLifecycle:
    @pytest.mark.asyncio
    async def test_setup_creates_console(self, sandbox):
        assert sandbox._console is not None

    @pytest.mark.asyncio
    async def test_health_when_setup(self, sandbox):
        assert await sandbox.health() is True

    @pytest.mark.asyncio
    async def test_health_when_not_setup(self):
        sb = LocalSandbox()
        assert await sb.health() is False

    @pytest.mark.asyncio
    async def test_teardown_clears_console(self, sandbox):
        await sandbox.teardown()
        assert sandbox._console is None
        assert await sandbox.health() is False


class TestLocalSandboxExecute:
    @pytest.mark.asyncio
    async def test_simple_expression(self, sandbox):
        result = json.loads(sandbox.execute("2 + 2"))
        assert result["success"] is True
        assert result["return_value"] == "4"

    @pytest.mark.asyncio
    async def test_print_captured(self, sandbox):
        result = json.loads(sandbox.execute("print('hello world')"))
        assert result["success"] is True
        assert "hello world" in result["stdout"]

    @pytest.mark.asyncio
    async def test_assignment_persists(self, sandbox):
        sandbox.execute("x = 42")
        result = json.loads(sandbox.execute("x + 8"))
        assert result["success"] is True
        assert result["return_value"] == "50"

    @pytest.mark.asyncio
    async def test_multiline_code(self, sandbox):
        code = "def add(a, b):\n    return a + b\nprint(add(3, 4))"
        result = json.loads(sandbox.execute(code))
        assert result["success"] is True
        assert "7" in result["stdout"]

    @pytest.mark.asyncio
    async def test_error_captured(self, sandbox):
        result = json.loads(sandbox.execute("1 / 0"))
        assert result["success"] is False
        assert "ZeroDivisionError" in result["stderr"]

    @pytest.mark.asyncio
    async def test_string_expression(self, sandbox):
        result = json.loads(sandbox.execute("'hello' + ' ' + 'world'"))
        assert result["success"] is True
        assert result["return_value"] == "'hello world'"


class TestLocalSandboxWhitelist:
    @pytest.mark.asyncio
    async def test_import_blocked(self, sandbox):
        result = json.loads(sandbox.execute("import os"))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_import_allowed(self, sandbox_with_modules):
        sandbox_with_modules.execute("import math")
        result = json.loads(sandbox_with_modules.execute("math.sqrt(16)"))
        assert result["success"] is True
        assert result["return_value"] == "4.0"

    @pytest.mark.asyncio
    async def test_import_not_in_whitelist(self, sandbox_with_modules):
        result = json.loads(sandbox_with_modules.execute("import os"))
        assert result["success"] is False
        assert "not allowed" in result["stderr"]

    @pytest.mark.asyncio
    async def test_import_submodule_allowed(self, sandbox_with_modules):
        """Importing json.decoder should work when json is in the whitelist."""
        sandbox_with_modules.execute("import json.decoder")
        result = json.loads(sandbox_with_modules.execute("json.decoder.JSONDecodeError"))
        assert result["success"] is True

    @pytest.mark.asyncio
    async def test_import_submodule_blocked(self, sandbox_with_modules):
        """Importing os.path should fail when os is not in the whitelist."""
        result = json.loads(sandbox_with_modules.execute("import os.path"))
        assert result["success"] is False
        assert "not allowed" in result["stderr"]

    @pytest.mark.asyncio
    async def test_builtin_open_blocked(self, sandbox):
        result = json.loads(sandbox.execute("open('/etc/passwd')"))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_safe_builtins_available(self, sandbox):
        result = json.loads(sandbox.execute("len([1, 2, 3])"))
        assert result["success"] is True
        assert result["return_value"] == "3"

    @pytest.mark.asyncio
    async def test_extra_builtin_allowed(self):
        sb = LocalSandbox(allowed_builtins=["ord"])
        await sb.setup()
        result = json.loads(sb.execute("ord('A')"))
        assert result["success"] is True
        assert result["return_value"] == "65"
        await sb.teardown()

    @pytest.mark.asyncio
    async def test_namespace_injection(self, sandbox_with_namespace):
        result = json.loads(sandbox_with_namespace.execute("len(data)"))
        assert result["success"] is True
        assert result["return_value"] == "3"

    @pytest.mark.asyncio
    async def test_namespace_injection_string(self, sandbox_with_namespace):
        result = json.loads(sandbox_with_namespace.execute("greeting"))
        assert result["success"] is True
        assert result["return_value"] == "'hello'"


class TestLocalSandboxReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self, sandbox):
        sandbox.execute("x = 42")
        sandbox.reset()
        result = json.loads(sandbox.execute("x"))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_reset_preserves_namespace(self):
        sb = LocalSandbox(namespace={"data": [1, 2, 3]})
        await sb.setup()
        sb.execute("data.append(4)")
        sb.reset()
        # data should be back to original
        result = json.loads(sb.execute("len(data)"))
        assert result["success"] is True
        assert result["return_value"] == "3"
        await sb.teardown()


# ---------------------------------------------------------------------------
# ProcessSandbox
# ---------------------------------------------------------------------------


@pytest.fixture
async def process_sandbox():
    """Create a basic ProcessSandbox."""
    sb = ProcessSandbox(timeout=5.0)
    await sb.setup()
    yield sb
    await sb.teardown()


@pytest.fixture
async def process_sandbox_with_modules():
    """Create a ProcessSandbox with math module allowed."""
    sb = ProcessSandbox(allowed_modules=["math"], timeout=5.0)
    await sb.setup()
    yield sb
    await sb.teardown()


class TestProcessSandboxLifecycle:
    @pytest.mark.asyncio
    async def test_setup_starts_process(self, process_sandbox):
        assert process_sandbox._process is not None
        assert process_sandbox._process.is_alive()

    @pytest.mark.asyncio
    async def test_health_when_running(self, process_sandbox):
        assert await process_sandbox.health() is True

    @pytest.mark.asyncio
    async def test_health_when_not_started(self):
        sb = ProcessSandbox()
        assert await sb.health() is False

    @pytest.mark.asyncio
    async def test_teardown_stops_process(self, process_sandbox):
        await process_sandbox.teardown()
        assert process_sandbox._process is None


class TestProcessSandboxExecute:
    @pytest.mark.asyncio
    async def test_simple_expression(self, process_sandbox):
        result = json.loads(process_sandbox.execute("2 + 2"))
        assert result["success"] is True
        assert result["return_value"] == "4"

    @pytest.mark.asyncio
    async def test_print_captured(self, process_sandbox):
        result = json.loads(process_sandbox.execute("print('hello')"))
        assert result["success"] is True
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_state_persists(self, process_sandbox):
        process_sandbox.execute("x = 10")
        result = json.loads(process_sandbox.execute("x * 2"))
        assert result["success"] is True
        assert result["return_value"] == "20"

    @pytest.mark.asyncio
    async def test_import_blocked(self, process_sandbox):
        result = json.loads(process_sandbox.execute("import os"))
        assert result["success"] is False

    @pytest.mark.asyncio
    async def test_import_allowed(self, process_sandbox_with_modules):
        process_sandbox_with_modules.execute("import math")
        result = json.loads(process_sandbox_with_modules.execute("math.pi"))
        assert result["success"] is True
        assert "3.14" in result["return_value"]

    @pytest.mark.asyncio
    async def test_timeout_kills_process(self):
        sb = ProcessSandbox(timeout=1.0)
        await sb.setup()
        try:
            result = json.loads(sb.execute("while True: pass"))
            assert result["success"] is False
            assert "timed out" in result["stderr"]
        finally:
            await sb.teardown()

    @pytest.mark.asyncio
    async def test_recovers_after_timeout(self):
        sb = ProcessSandbox(timeout=1.0)
        await sb.setup()
        try:
            sb.execute("while True: pass")  # times out
            # Should still work after restart
            result = json.loads(sb.execute("1 + 1"))
            assert result["success"] is True
            assert result["return_value"] == "2"
        finally:
            await sb.teardown()


class TestProcessSandboxReset:
    @pytest.mark.asyncio
    async def test_reset_clears_state(self, process_sandbox):
        process_sandbox.execute("x = 42")
        process_sandbox.reset()
        result = json.loads(process_sandbox.execute("x"))
        assert result["success"] is False


# ---------------------------------------------------------------------------
# Tool integration
# ---------------------------------------------------------------------------


class TestSandboxToolIntegration:
    def test_local_sandbox_get_tools_prefixed(self):
        sb = LocalSandbox()
        tools = sb.get_tools()
        names = [t.__name__ for t in tools]
        assert "local_sandbox__execute" in names
        assert "local_sandbox__reset" in names

    def test_process_sandbox_get_tools_prefixed(self):
        sb = ProcessSandbox()
        tools = sb.get_tools()
        names = [t.__name__ for t in tools]
        assert "process_sandbox__execute" in names
        assert "process_sandbox__reset" in names

    def test_tools_have_schemas(self):
        sb = LocalSandbox()
        tools = sb.get_tools()
        for t in tools:
            assert hasattr(t, "_tool_schema")
            schema = t._tool_schema
            func_schema = schema.get("function", schema)
            assert func_schema["name"].startswith("local_sandbox__")
