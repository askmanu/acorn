"""Sandbox services for code execution with whitelisted namespaces.

.. warning::

    These sandboxes are NOT safe for production use with untrusted code.
    LocalSandbox and ProcessSandbox use namespace restriction which can be
    bypassed by a determined attacker (e.g. via ``object.__subclasses__()``).
    DockerSandbox provides stronger isolation but requires careful configuration.
    For production use with untrusted input, use a dedicated sandboxing service
    such as E2B, Daytona, or gVisor-backed containers.
"""

import builtins
import code
import copy
import io
import json
import multiprocessing
import sys
import threading
from typing import Optional

from acorn.decorators import tool
from acorn.service import Service


# Builtins that are always available in the sandbox
_SAFE_BUILTINS = {
    "print", "len", "range", "list", "dict", "set", "tuple",
    "str", "int", "float", "bool", "type",
    "True", "False", "None",
    "abs", "min", "max", "sum", "sorted", "reversed",
    "enumerate", "zip", "map", "filter",
    "isinstance", "issubclass", "hasattr", "getattr",
    "round", "divmod", "pow",
    "repr", "format", "hash",
    "iter", "next", "all", "any",
}


def _build_restricted_builtins(extra_builtins: Optional[list[str]] = None) -> dict:
    """Build a restricted __builtins__ dict from the safe set plus extras."""
    allowed = _SAFE_BUILTINS | set(extra_builtins or [])
    return {
        name: getattr(builtins, name)
        for name in allowed
        if hasattr(builtins, name)
    }


def _build_namespace(
    allowed_modules: Optional[list[str]] = None,
    allowed_builtins: Optional[list[str]] = None,
    namespace: Optional[dict] = None,
) -> dict:
    """Build a complete sandbox namespace with restricted builtins.

    Args:
        allowed_modules: Module names the sandbox can import. None means
                         no imports allowed.
        allowed_builtins: Additional builtin names beyond the safe default.
        namespace: Extra objects to inject into the namespace.
    """
    restricted = _build_restricted_builtins(allowed_builtins)

    if allowed_modules:
        allowed_set = set(allowed_modules)
        real_import = builtins.__import__

        def restricted_import(name, *args, **kwargs):
            # Check top-level package so "import json.decoder" works
            # when "json" is in the allowed set
            top_level = name.split(".")[0]
            if top_level not in allowed_set:
                raise ImportError(f"Import of '{name}' is not allowed")
            return real_import(name, *args, **kwargs)

        restricted["__import__"] = restricted_import

    ns = {"__builtins__": restricted}
    if namespace:
        ns.update(namespace)
    return ns


class _CapturedConsole(code.InteractiveConsole):
    """InteractiveConsole that captures output and returns structured results.

    Note: stdout/stderr redirection uses a threading lock to prevent
    interleaving when multiple sandboxes execute concurrently, but this
    only protects against other _CapturedConsole instances — any code
    outside the sandbox writing to sys.stdout during execution may still
    have its output swallowed.
    """

    _io_lock = threading.Lock()

    def __init__(self, locals=None):
        super().__init__(locals=locals)
        self._stderr_buf = io.StringIO()

    def write(self, data):
        """Capture traceback/error output (called by InteractiveConsole)."""
        self._stderr_buf.write(data)

    def push_and_capture(self, source: str) -> dict:
        """Execute source and return captured output.

        Returns:
            Dict with stdout, stderr, return_value, and success fields.
        """
        stdout_buf = io.StringIO()
        self._stderr_buf = io.StringIO()
        return_value = None
        success = True

        with self._io_lock:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = stdout_buf
            sys.stderr = self._stderr_buf
            try:
                # Try as expression first to capture return value
                try:
                    compiled = compile(source, "<sandbox>", "eval")
                    result = eval(compiled, self.locals)  # noqa: S307
                    if result is not None:
                        return_value = repr(result)
                except SyntaxError:
                    # Statement or multi-line — use exec
                    exec(compile(source, "<sandbox>", "exec"), self.locals)  # noqa: S102
                    return_value = "None"
            except Exception as e:
                self._stderr_buf.write(f"{type(e).__name__}: {e}\n")
                return_value = None
                success = False
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

        return {
            "stdout": stdout_buf.getvalue(),
            "stderr": self._stderr_buf.getvalue(),
            "return_value": return_value,
            "success": success,
        }


class Sandbox(Service):
    """Base class for code execution sandboxes.

    Subclasses implement execute() and reset() with different isolation
    levels: in-process namespace restriction, subprocess isolation, or
    full container isolation.

    .. warning::

        Not safe for production use with untrusted code. Use a dedicated
        sandboxing service (E2B, Daytona, gVisor) for adversarial inputs.
    """

    @tool
    def execute(self, code: str) -> str:
        """Execute Python code in the sandbox.

        Args:
            code: Python code to execute
        """
        raise NotImplementedError

    @tool
    def reset(self) -> str:
        """Reset the sandbox, clearing all variables and state."""
        raise NotImplementedError


class LocalSandbox(Sandbox):
    """In-process Python code execution with whitelisted namespace.

    Uses code.InteractiveConsole with restricted builtins. State persists
    across calls within the same session. No timeout support — use
    ProcessSandbox if you need timeouts.

    .. warning::

        Namespace restriction is NOT a security sandbox. A determined attacker
        can bypass it (e.g. via ``object.__subclasses__()``). Use this only
        for agent-generated code where you control the system prompt.
        stdout/stderr capture redirects global sys.stdout — other coroutines
        writing to stdout during execution may have output swallowed.
    """

    def __init__(
        self,
        allowed_modules: Optional[list[str]] = None,
        allowed_builtins: Optional[list[str]] = None,
        namespace: Optional[dict] = None,
    ):
        """Initialize the LocalSandbox.

        Args:
            allowed_modules: Module names the sandbox can import (e.g. ["math", "json"]).
                           Default: no imports allowed.
            allowed_builtins: Builtin names to expose beyond the safe default set.
            namespace: Additional objects to inject into the sandbox namespace.
        """
        self._allowed_modules = allowed_modules
        self._allowed_builtins = allowed_builtins
        self._extra_namespace = namespace
        self._console: Optional[_CapturedConsole] = None
        self._initial_namespace: Optional[dict] = None

    async def setup(self):
        """Create the sandbox console with restricted namespace."""
        self._initial_namespace = _build_namespace(
            self._allowed_modules, self._allowed_builtins, self._extra_namespace
        )
        self._console = _CapturedConsole(locals=copy.deepcopy(self._initial_namespace))

    async def teardown(self):
        """Destroy the sandbox console."""
        self._console = None
        self._initial_namespace = None

    async def health(self) -> bool:
        """Check if the sandbox console is active."""
        return self._console is not None

    @tool
    def execute(self, code: str) -> str:
        """Execute Python code in the sandbox and return the result.

        Args:
            code: Python code to execute
        """
        result = self._console.push_and_capture(code)
        return json.dumps(result)

    @tool
    def reset(self) -> str:
        """Reset the sandbox, clearing all variables and state."""
        self._console = _CapturedConsole(locals=copy.deepcopy(self._initial_namespace))
        return "Sandbox state reset."


def _sandbox_worker(conn, namespace_args):
    """Worker function that runs in a child process.

    Receives code strings via the pipe, executes them in a CapturedConsole,
    and sends result dicts back. Sentinel values:
      None        -> shutdown
      "__reset__" -> rebuild console with fresh namespace
    """
    namespace = _build_namespace(*namespace_args)
    console = _CapturedConsole(locals=copy.deepcopy(namespace))

    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break

        if msg is None:
            break

        if msg == "__reset__":
            console = _CapturedConsole(locals=copy.deepcopy(namespace))
            conn.send({"stdout": "", "stderr": "", "return_value": "None", "success": True})
            continue

        result = console.push_and_capture(msg)
        conn.send(result)

    conn.close()


class ProcessSandbox(Sandbox):
    """Code execution in a child process with timeout and kill support.

    Same whitelist model as LocalSandbox but runs in a subprocess via
    multiprocessing. Supports hard timeout (process is killed). State
    persists across calls but is lost if the process is killed on timeout.
    """

    def __init__(
        self,
        allowed_modules: Optional[list[str]] = None,
        allowed_builtins: Optional[list[str]] = None,
        namespace: Optional[dict] = None,
        timeout: float = 30.0,
    ):
        """Initialize the ProcessSandbox.

        Args:
            allowed_modules: Module names the sandbox can import.
            allowed_builtins: Builtin names to expose beyond the safe default set.
            namespace: Additional objects to inject into the sandbox namespace.
            timeout: Maximum execution time in seconds per execute() call.
        """
        self._allowed_modules = allowed_modules
        self._allowed_builtins = allowed_builtins
        self._extra_namespace = namespace
        self._timeout = timeout
        self._process: Optional[multiprocessing.Process] = None
        self._conn = None

    def _namespace_args(self):
        """Return args tuple for _build_namespace (serializable across processes)."""
        return (self._allowed_modules, self._allowed_builtins, self._extra_namespace)

    def _start_process(self):
        """Start (or restart) the worker process."""
        parent_conn, child_conn = multiprocessing.Pipe()
        process = multiprocessing.Process(
            target=_sandbox_worker,
            args=(child_conn, self._namespace_args()),
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._conn = parent_conn
        self._process = process

    async def setup(self):
        """Start the sandbox worker process."""
        self._start_process()

    async def teardown(self):
        """Shut down the sandbox worker process."""
        if self._conn:
            try:
                self._conn.send(None)
                self._conn.close()
            except (OSError, BrokenPipeError):
                pass
            self._conn = None
        if self._process and self._process.is_alive():
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(timeout=2)
        self._process = None

    async def health(self) -> bool:
        """Check if the worker process is alive."""
        return self._process is not None and self._process.is_alive()

    @tool
    def execute(self, code: str) -> str:
        """Execute Python code in the sandbox and return the result.

        Args:
            code: Python code to execute
        """
        try:
            self._conn.send(code)
        except (OSError, BrokenPipeError):
            return json.dumps({
                "stdout": "",
                "stderr": "Sandbox process is not running.",
                "return_value": None,
                "success": False,
            })

        if self._conn.poll(timeout=self._timeout):
            result = self._conn.recv()
            return json.dumps(result)
        else:
            # Timeout — kill and restart
            self._process.kill()
            self._process.join(timeout=5)
            self._start_process()
            return json.dumps({
                "stdout": "",
                "stderr": f"Execution timed out after {self._timeout}s. Sandbox has been reset.",
                "return_value": None,
                "success": False,
            })

    @tool
    def reset(self) -> str:
        """Reset the sandbox, clearing all variables and state."""
        try:
            self._conn.send("__reset__")
            if self._conn.poll(timeout=5):
                self._conn.recv()
                return "Sandbox state reset."
            else:
                # Reset timed out — restart process
                self._process.kill()
                self._process.join(timeout=5)
                self._start_process()
                return "Sandbox process restarted."
        except (OSError, BrokenPipeError):
            self._start_process()
            return "Sandbox process restarted."
