"""Tests for module initialization and configuration."""

import pytest
from pydantic import BaseModel
from acorn import Module, tool
from acorn.exceptions import ToolConflictError


def test_module_instantiation():
    """Test basic module instantiation."""
    class SimpleModule(Module):
        pass

    mod = SimpleModule()
    assert mod is not None
    assert mod.model == "anthropic/claude-sonnet-4-5-20250514"
    assert mod.temperature == 0.7
    assert mod.max_tokens == 4096


def test_module_custom_config():
    """Test module with custom configuration."""
    class CustomModule(Module):
        model = "gpt-4"
        temperature = 0.5
        max_tokens = 2000

    mod = CustomModule()
    assert mod.model == "gpt-4"
    assert mod.temperature == 0.5
    assert mod.max_tokens == 2000


def test_module_collects_tools_from_list():
    """Test that module collects tools from tools list."""
    def external_tool(x: int) -> int:
        """External tool."""
        return x * 2

    class ModuleWithTools(Module):
        tools = [external_tool]

    mod = ModuleWithTools()
    assert len(mod._collected_tools) == 1
    assert mod._collected_tools[0] is external_tool


def test_module_collects_tools_from_methods():
    """Test that module collects @tool decorated methods."""
    class ModuleWithMethods(Module):
        @tool
        def my_tool(self, x: int) -> int:
            """My tool."""
            return x + 1

    mod = ModuleWithMethods()
    assert len(mod._collected_tools) == 1
    assert mod._collected_tools[0].__name__ == "my_tool"


def test_module_collects_both_tool_types():
    """Test collecting tools from both list and methods."""
    def external_tool(x: int) -> int:
        """External."""
        return x

    class MixedModule(Module):
        tools = [external_tool]

        @tool
        def internal_tool(self, y: int) -> int:
            """Internal."""
            return y

    mod = MixedModule()
    assert len(mod._collected_tools) == 2


def test_tool_conflict_detection():
    """Test that duplicate tool names raise ToolConflictError."""
    @tool
    def my_tool(x: int) -> int:
        """Tool one."""
        return x

    @tool
    def my_tool_copy(x: int) -> int:
        """Tool two with same name after decoration."""
        return x

    # Manually set same name to create conflict
    my_tool_copy.__name__ = "my_tool"

    class ConflictModule(Module):
        tools = [my_tool, my_tool_copy]

    with pytest.raises(ToolConflictError, match="Duplicate tool name"):
        ConflictModule()


def test_system_prompt_from_string():
    """Test system prompt from string."""
    class StringPromptModule(Module):
        system_prompt = "You are a helpful assistant."

    mod = StringPromptModule()
    msg = mod._build_system_message()
    assert msg["role"] == "system"
    assert msg["content"] == "You are a helpful assistant."


def test_system_prompt_from_docstring():
    """Test system prompt from class docstring."""
    class DocstringModule(Module):
        """This is the system prompt from docstring."""

    mod = DocstringModule()
    msg = mod._build_system_message()
    assert "This is the system prompt from docstring" in msg["content"]


def test_system_prompt_empty():
    """Test empty system prompt."""
    class NoPromptModule(Module):
        pass

    mod = NoPromptModule()
    msg = mod._build_system_message()
    assert msg["content"] == ""


def test_schemas_defined():
    """Test module with input and output schemas."""
    class Input(BaseModel):
        text: str

    class Output(BaseModel):
        result: str

    class SchemaModule(Module):
        initial_input = Input
        final_output = Output

    mod = SchemaModule()
    assert mod.initial_input is Input
    assert mod.final_output is Output


def test_finish_tool_generation():
    """Test __finish__ tool generation."""
    class Output(BaseModel):
        summary: str
        score: int

    class FinishModule(Module):
        final_output = Output

    mod = FinishModule()
    finish_tool = mod._generate_finish_tool()

    assert hasattr(finish_tool, "_tool_schema")
    schema = finish_tool._tool_schema
    assert schema["function"]["name"] == "__finish__"
    assert "summary" in schema["function"]["parameters"]["properties"]
    assert "score" in schema["function"]["parameters"]["properties"]


def test_single_turn_mode():
    """Test that max_steps=None means single-turn."""
    class SingleTurnModule(Module):
        max_steps = None

    mod = SingleTurnModule()
    assert mod.max_steps is None
