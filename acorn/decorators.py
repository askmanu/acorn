"""Decorators for Acorn."""

from typing import Callable
from acorn.tool_schema import generate_tool_schema


def tool(func: Callable) -> Callable:
    """Mark a function as a tool with automatic schema generation.

    The @tool decorator generates a JSON schema from the function's
    signature and docstring, making it available to LLMs for tool calling.

    Args:
        func: The function to mark as a tool

    Returns:
        The function with added _tool_schema attribute

    Example:
        >>> @tool
        ... def search(query: str, limit: int = 10) -> list:
        ...     '''Search for items.
        ...
        ...     Args:
        ...         query: The search query
        ...         limit: Maximum number of results
        ...     '''
        ...     return []
        >>> hasattr(search, '_tool_schema')
        True
    """
    # Generate schema
    schema = generate_tool_schema(func)

    # Attach to function
    func._tool_schema = schema

    return func
