"""Generate JSON schemas for tools from Python functions."""

import inspect
import re
from typing import get_type_hints, get_origin, get_args, Any, Callable


def generate_tool_schema(func: Callable) -> dict:
    """Generate JSON schema for a tool from its function signature.

    Args:
        func: A Python function (may be decorated with @tool)

    Returns:
        JSON schema dictionary compatible with LiteLLM/OpenAI tool format

    Example:
        >>> def search(query: str, limit: int = 10) -> list:
        ...     '''Search for items.
        ...
        ...     Args:
        ...         query: The search query
        ...         limit: Maximum number of results
        ...     '''
        ...     pass
        >>> schema = generate_tool_schema(search)
        >>> schema['function']['name']
        'search'
    """
    # Get function name
    name = func.__name__

    # Get description from docstring
    description = _extract_description(func.__doc__ or "")

    # Get function signature
    sig = inspect.signature(func)

    # Get type hints
    try:
        type_hints = get_type_hints(func)
    except:
        type_hints = {}

    # Parse parameter descriptions from docstring
    param_descriptions = _parse_docstring_params(func.__doc__ or "")

    # Build parameters schema
    parameters = {
        "type": "object",
        "properties": {},
        "required": []
    }

    for param_name, param in sig.parameters.items():
        # Skip 'self' parameter for methods
        if param_name == "self":
            continue

        # Get type hint - try type_hints first, then param.annotation
        if param_name in type_hints:
            param_type = type_hints[param_name]
        elif param.annotation != inspect.Parameter.empty:
            param_type = param.annotation
        else:
            param_type = Any

        # Convert Python type to JSON schema type
        param_schema = _python_type_to_json_schema(param_type)

        # Add description if available
        if param_name in param_descriptions:
            param_schema["description"] = param_descriptions[param_name]

        parameters["properties"][param_name] = param_schema

        # Add to required if no default value
        if param.default == inspect.Parameter.empty:
            parameters["required"].append(param_name)

    # Build full schema
    schema = {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters
        }
    }

    return schema


def _extract_description(docstring: str) -> str:
    """Extract the main description from a docstring.

    Takes the first paragraph before Args/Returns/etc sections.
    """
    if not docstring:
        return ""

    # Split into lines and clean
    lines = [line.strip() for line in docstring.strip().split("\n")]

    # Find the first paragraph (before Args, Returns, etc.)
    description_lines = []
    for line in lines:
        # Stop at section headers
        if line.startswith(("Args:", "Arguments:", "Parameters:", "Returns:", "Raises:", "Example:")):
            break
        if line:
            description_lines.append(line)

    return " ".join(description_lines)


def _parse_docstring_params(docstring: str) -> dict[str, str]:
    """Parse parameter descriptions from docstring.

    Supports multiple docstring formats (Google, NumPy, reStructuredText).

    Returns:
        Dict mapping parameter names to descriptions
    """
    if not docstring:
        return {}

    param_descriptions = {}

    # Find Args/Arguments/Parameters section
    lines = docstring.split("\n")
    in_args_section = False
    current_param = None
    current_desc = []

    for line in lines:
        stripped = line.strip()

        # Check for Args section start
        if stripped in ("Args:", "Arguments:", "Parameters:"):
            in_args_section = True
            continue

        # Check for end of Args section (new section starts)
        if in_args_section and stripped.endswith(":") and not stripped.startswith(" "):
            break

        if in_args_section:
            # Check if this is a parameter line (has colon)
            # Format: "param_name: description" or "param_name (type): description"
            match = re.match(r"^\s*(\w+)(?:\s*\([^)]+\))?\s*:\s*(.+)", line)
            if match:
                # Save previous parameter
                if current_param:
                    param_descriptions[current_param] = " ".join(current_desc).strip()

                # Start new parameter
                current_param = match.group(1)
                current_desc = [match.group(2)]
            elif current_param and stripped:
                # Continuation of previous parameter description
                current_desc.append(stripped)

    # Save last parameter
    if current_param:
        param_descriptions[current_param] = " ".join(current_desc).strip()

    return param_descriptions


def _python_type_to_json_schema(python_type: type) -> dict:
    """Convert a Python type hint to JSON schema.

    Args:
        python_type: A Python type from type hints

    Returns:
        JSON schema dictionary for the type
    """
    # Handle None/NoneType
    if python_type is None or python_type == type(None):
        return {"type": "null"}

    # Get origin for generic types
    origin = get_origin(python_type)

    # Handle Union types (including Optional and | syntax)
    if origin is not None:
        # Check for UnionType (Python 3.10+ | syntax)
        origin_str = str(origin)
        if 'Union' in origin_str or 'UnionType' in origin_str:
            args = get_args(python_type)
            if args:
                # Get non-None types
                non_none_types = [arg for arg in args if arg is not type(None)]
                if len(non_none_types) == 1:
                    # Optional[X] or X | None - just use X schema
                    return _python_type_to_json_schema(non_none_types[0])
                elif len(non_none_types) > 1:
                    # Union of multiple types
                    return {"anyOf": [_python_type_to_json_schema(t) for t in non_none_types]}

    # Handle list (both list[X] and bare list)
    if origin is list or python_type is list or python_type == list:
        args = get_args(python_type)
        if args:
            return {
                "type": "array",
                "items": _python_type_to_json_schema(args[0])
            }
        else:
            return {"type": "array"}

    # Handle dict (both dict[K, V] and bare dict)
    if origin is dict or python_type is dict or python_type == dict:
        args = get_args(python_type)
        if len(args) >= 2:
            # Dict[str, ValueType]
            return {
                "type": "object",
                "additionalProperties": _python_type_to_json_schema(args[1])
            }
        else:
            return {"type": "object"}

    # Handle basic types
    if python_type == str or python_type is str:
        return {"type": "string"}
    elif python_type == int or python_type is int:
        return {"type": "integer"}
    elif python_type == float or python_type is float:
        return {"type": "number"}
    elif python_type == bool or python_type is bool:
        return {"type": "boolean"}
    elif python_type == Any:
        return {}  # No type restriction

    # Default fallback
    return {"type": "string"}
