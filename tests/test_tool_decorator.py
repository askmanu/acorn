"""Tests for @tool decorator."""

import pytest
from acorn.decorators import tool


def test_tool_decorator_basic():
    """Test basic @tool decorator usage."""
    @tool
    def my_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    # Function should still work
    assert my_tool(5) == 10

    # Should have schema attached
    assert hasattr(my_tool, '_tool_schema')
    assert isinstance(my_tool._tool_schema, dict)


def test_tool_schema_content():
    """Test that @tool generates correct schema."""
    @tool
    def search(query: str, limit: int = 10) -> list:
        """Search for items.

        Args:
            query: The search query
            limit: Maximum results
        """
        return []

    schema = search._tool_schema

    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search"
    assert "query" in schema["function"]["parameters"]["properties"]


def test_tool_preserves_function():
    """Test that @tool preserves function behavior."""
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    # Should still work normally
    assert add(2, 3) == 5
    assert add.__name__ == "add"
    assert "Add two numbers" in add.__doc__


def test_tool_on_method():
    """Test @tool on class methods."""
    class Calculator:
        @tool
        def multiply(self, a: int, b: int) -> int:
            """Multiply two numbers.

            Args:
                a: First number
                b: Second number
            """
            return a * b

    calc = Calculator()

    # Method should work
    assert calc.multiply(3, 4) == 12

    # Should have schema (excluding self)
    schema = calc.multiply._tool_schema
    assert "self" not in schema["function"]["parameters"]["properties"]
    assert "a" in schema["function"]["parameters"]["properties"]


def test_multiple_tools():
    """Test decorating multiple functions."""
    @tool
    def tool1(x: int) -> int:
        """Tool one."""
        return x

    @tool
    def tool2(y: str) -> str:
        """Tool two."""
        return y

    # Each should have its own schema
    assert tool1._tool_schema["function"]["name"] == "tool1"
    assert tool2._tool_schema["function"]["name"] == "tool2"
    assert tool1._tool_schema is not tool2._tool_schema


def test_tool_without_type_hints():
    """Test @tool on function without type hints."""
    @tool
    def no_types(x):
        """A function without types."""
        return x

    # Should still generate schema, but with generic types
    assert hasattr(no_types, '_tool_schema')
    schema = no_types._tool_schema
    assert "x" in schema["function"]["parameters"]["properties"]


def test_tool_with_complex_signature():
    """Test @tool with complex function signature."""
    @tool
    def complex_tool(
        required: str,
        optional: int = 5,
        *args,
        **kwargs
    ) -> dict:
        """Complex signature.

        Args:
            required: A required param
            optional: An optional param
        """
        return {}

    schema = complex_tool._tool_schema

    # Should handle required and optional params
    assert "required" in schema["function"]["parameters"]["required"]
    assert "optional" not in schema["function"]["parameters"]["required"]
