"""Service base class for Acorn tool collections."""

import copy
import functools
import re
from typing import Any

from acorn.tool_schema import generate_tool_schema


def _to_snake_case(name: str) -> str:
    """Convert CamelCase or PascalCase to snake_case.

    Examples:
        Gmail -> gmail
        GoogleCalendar -> google_calendar
        E2BSandbox -> e2b_sandbox
        Memory -> memory
        HTTPClient -> http_client
    """
    # Extract word tokens by matching:
    # 1. Acronym+digits before a capitalized word (e.g., "HTTP" in "HTTPClient")
    # 2. Terminal acronym+digits not followed by lowercase (e.g., "API" at end)
    # 3. Capitalized word (e.g., "Client", "Gmail")
    # 4. Lowercase word (e.g., "memory")
    tokens = re.findall(
        r"[A-Z0-9]+(?=[A-Z][a-z])|[A-Z0-9]+(?![a-z])|[A-Z][a-z0-9]*|[a-z][a-z0-9]*",
        name,
    )
    return "_".join(tokens).lower()


def _prefix_tool(tool_func: Any, prefix: str) -> Any:
    """Create a prefixed wrapper around a tool function.

    The wrapper has __name__ = '{prefix}__{original_name}' and a cloned
    _tool_schema with the prefixed name. The original function is called
    unchanged.

    Args:
        tool_func: A callable with _tool_schema attribute
        prefix: The service prefix (e.g., 'gmail')

    Returns:
        A wrapper callable with prefixed name and schema
    """
    original_name = tool_func.__name__
    prefixed_name = f"{prefix}__{original_name}"

    @functools.wraps(tool_func)
    def wrapper(*args, **kwargs):
        return tool_func(*args, **kwargs)

    # Set the prefixed name
    wrapper.__name__ = prefixed_name

    # Clone and update the tool schema
    schema = copy.deepcopy(tool_func._tool_schema)
    if "function" in schema:
        schema["function"]["name"] = prefixed_name
    elif "name" in schema:
        schema["name"] = prefixed_name
    wrapper._tool_schema = schema

    return wrapper


class Service:
    """Base class for tool collections with shared config and lifecycle.

    A Service groups related tools that share configuration (API keys,
    database connections, etc.) and provides lifecycle management.

    Class name is used as the service name, docstring as the description.
    Tool methods decorated with @tool are auto-collected and prefixed with
    the service name (e.g., Gmail.send -> gmail__send).

    Example:
        >>> from acorn import Service, tool
        >>> class Gmail(Service):
        ...     '''Google Gmail integration.'''
        ...     def __init__(self, token: str):
        ...         self.token = token
        ...
        ...     @tool
        ...     def send(self, to: str, subject: str, body: str) -> str:
        ...         '''Send an email.'''
        ...         return f"Sent to {to}"
        ...
        ...     @tool
        ...     def search(self, query: str) -> str:
        ...         '''Search emails.'''
        ...         return f"Results for {query}"
        >>> gmail = Gmail(token="...")
        >>> tools = gmail.get_tools()
        >>> [t.__name__ for t in tools]
        ['gmail__send', 'gmail__search']
    """

    @property
    def name(self) -> str:
        """Service name, derived from class name."""
        return self.__class__.__name__

    @property
    def description(self) -> str:
        """Service description, derived from class docstring."""
        return self.__doc__ or ""

    async def setup(self):
        """Called when the module starts. Override for initialization.

        Use for establishing database connections, refreshing auth tokens,
        or any async setup that can't happen in __init__.
        """
        pass

    async def teardown(self):
        """Called when the module finishes. Override for cleanup.

        Use for closing database connections, flushing buffers, etc.
        """
        pass

    async def health(self) -> bool:
        """Check if the service is operational.

        Returns:
            True if the service is healthy, False otherwise.
        """
        return True

    def get_tools(self) -> list:
        """Collect all @tool-decorated methods, with auto-prefixed names.

        Tools are prefixed with snake_case(service_name)__ to prevent
        conflicts when multiple services are used together.

        Returns:
            List of prefixed tool callables with _tool_schema attributes.
        """
        tools = []
        prefix = _to_snake_case(self.name)

        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue

            attr = getattr(self, attr_name)

            if callable(attr) and hasattr(attr, "_tool_schema"):
                prefixed = _prefix_tool(attr, prefix)
                tools.append(prefixed)

        return tools

    async def __aenter__(self):
        """Context manager entry — calls setup()."""
        await self.setup()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit — calls teardown()."""
        await self.teardown()
        return False
