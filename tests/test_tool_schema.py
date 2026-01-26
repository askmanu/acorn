"""Tests for tool schema generation."""

import pytest
from typing import Optional
from acorn.tool_schema import generate_tool_schema


def test_simple_function():
    """Test schema generation for simple function."""
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    schema = generate_tool_schema(add)

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "add"
    assert schema["function"]["description"] == "Add two numbers."
    assert "a" in schema["function"]["parameters"]["properties"]
    assert "b" in schema["function"]["parameters"]["properties"]
    assert schema["function"]["parameters"]["properties"]["a"]["type"] == "integer"
    assert schema["function"]["parameters"]["properties"]["b"]["type"] == "integer"
    assert set(schema["function"]["parameters"]["required"]) == {"a", "b"}


def test_function_with_defaults():
    """Test schema generation with default values."""
    def search(query: str, limit: int = 10) -> list:
        """Search for items."""
        return []

    schema = generate_tool_schema(search)

    assert "query" in schema["function"]["parameters"]["properties"]
    assert "limit" in schema["function"]["parameters"]["properties"]
    assert "query" in schema["function"]["parameters"]["required"]
    assert "limit" not in schema["function"]["parameters"]["required"]


def test_function_with_optional():
    """Test schema generation with Optional types."""
    def process(data: str, metadata: Optional[dict] = None) -> bool:
        """Process data."""
        return True

    schema = generate_tool_schema(process)

    assert "data" in schema["function"]["parameters"]["properties"]
    assert "metadata" in schema["function"]["parameters"]["properties"]
    assert "data" in schema["function"]["parameters"]["required"]
    assert "metadata" not in schema["function"]["parameters"]["required"]
    # Optional[dict] should be unwrapped to just dict
    assert schema["function"]["parameters"]["properties"]["metadata"]["type"] == "object"


def test_function_with_list_type():
    """Test schema generation with list types."""
    def batch_process(items: list[str]) -> int:
        """Process multiple items."""
        return len(items)

    schema = generate_tool_schema(batch_process)

    assert schema["function"]["parameters"]["properties"]["items"]["type"] == "array"
    assert schema["function"]["parameters"]["properties"]["items"]["items"]["type"] == "string"


def test_function_with_dict_type():
    """Test schema generation with dict types."""
    def configure(settings: dict[str, int]) -> bool:
        """Configure settings."""
        return True

    schema = generate_tool_schema(configure)

    assert schema["function"]["parameters"]["properties"]["settings"]["type"] == "object"
    assert schema["function"]["parameters"]["properties"]["settings"]["additionalProperties"]["type"] == "integer"


def test_docstring_parameter_descriptions():
    """Test parsing parameter descriptions from docstring."""
    def search(query: str, limit: int, offset: int) -> list:
        """Search for items.

        Args:
            query: The search query string
            limit: Maximum number of results to return
            offset: Number of results to skip
        """
        return []

    schema = generate_tool_schema(search)

    assert schema["function"]["parameters"]["properties"]["query"]["description"] == "The search query string"
    assert schema["function"]["parameters"]["properties"]["limit"]["description"] == "Maximum number of results to return"
    assert schema["function"]["parameters"]["properties"]["offset"]["description"] == "Number of results to skip"


def test_docstring_multiline_descriptions():
    """Test parsing multiline parameter descriptions."""
    def complex_func(param1: str) -> None:
        """Do something complex.

        Args:
            param1: This is a long description
                that spans multiple lines
                and should be joined together
        """
        pass

    schema = generate_tool_schema(complex_func)

    desc = schema["function"]["parameters"]["properties"]["param1"]["description"]
    assert "This is a long description that spans multiple lines and should be joined together" == desc


def test_docstring_with_type_hints():
    """Test parsing parameters with type hints in docstring."""
    def typed_func(name: str, count: int) -> None:
        """Do something.

        Args:
            name (str): The name
            count (int): The count
        """
        pass

    schema = generate_tool_schema(typed_func)

    assert schema["function"]["parameters"]["properties"]["name"]["description"] == "The name"
    assert schema["function"]["parameters"]["properties"]["count"]["description"] == "The count"


def test_method_excludes_self():
    """Test that self parameter is excluded for methods."""
    class MyClass:
        def my_method(self, param: str) -> bool:
            """Do something."""
            return True

    obj = MyClass()
    schema = generate_tool_schema(obj.my_method)

    # Should not include 'self'
    assert "self" not in schema["function"]["parameters"]["properties"]
    assert "param" in schema["function"]["parameters"]["properties"]


def test_no_docstring():
    """Test function without docstring."""
    def no_doc(x: int) -> int:
        return x * 2

    schema = generate_tool_schema(no_doc)

    assert schema["function"]["name"] == "no_doc"
    assert schema["function"]["description"] == ""


def test_various_type_hints():
    """Test various Python types."""
    def all_types(
        s: str,
        i: int,
        f: float,
        b: bool,
        lst: list,
        dct: dict,
    ) -> None:
        """Test all types."""
        pass

    schema = generate_tool_schema(all_types)

    props = schema["function"]["parameters"]["properties"]
    assert props["s"]["type"] == "string"
    assert props["i"]["type"] == "integer"
    assert props["f"]["type"] == "number"
    assert props["b"]["type"] == "boolean"
    assert props["lst"]["type"] == "array"
    assert props["dct"]["type"] == "object"


def test_no_parameters():
    """Test function with no parameters."""
    def no_params() -> str:
        """Get something."""
        return "result"

    schema = generate_tool_schema(no_params)

    assert schema["function"]["parameters"]["properties"] == {}
    assert schema["function"]["parameters"]["required"] == []


def test_union_type():
    """Test Union type handling."""
    def union_param(value: str | int) -> None:
        """Handle union type."""
        pass

    schema = generate_tool_schema(union_param)

    # Union should be handled as anyOf
    prop = schema["function"]["parameters"]["properties"]["value"]
    assert "anyOf" in prop
    types = [t["type"] for t in prop["anyOf"]]
    assert set(types) == {"string", "integer"}
