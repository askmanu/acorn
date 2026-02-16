"""Tests for module initialization and configuration."""

import pytest
from pydantic import BaseModel
from acorn import Module, tool
from acorn.exceptions import ToolConflictError


def test_module_instantiation():
    """Test basic module instantiation."""
    class Output(BaseModel):
        result: str

    class SimpleModule(Module):
        model = "test-model"
        final_output = Output

    mod = SimpleModule()
    assert mod is not None
    assert mod.model == "test-model"
    assert mod.temperature == 0.7
    assert mod.max_tokens == 4096


def test_module_custom_config():
    """Test module with custom configuration."""
    class Output(BaseModel):
        result: str

    class CustomModule(Module):
        model = "gpt-4"
        temperature = 0.5
        max_tokens = 2000
        final_output = Output

    mod = CustomModule()
    assert mod.model == "gpt-4"
    assert mod.temperature == 0.5
    assert mod.max_tokens == 2000


def test_module_collects_tools_from_list():
    """Test that module collects tools from tools list."""
    def external_tool(x: int) -> int:
        """External tool."""
        return x * 2

    class Output(BaseModel):
        result: str

    class ModuleWithTools(Module):
        model = "test-model"
        tools = [external_tool]
        final_output = Output

    mod = ModuleWithTools()
    assert len(mod._collected_tools) == 1
    assert mod._collected_tools[0] is external_tool


def test_module_collects_tools_from_methods():
    """Test that module collects @tool decorated methods."""
    class Output(BaseModel):
        result: str

    class ModuleWithMethods(Module):
        model = "test-model"
        final_output = Output

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

    class Output(BaseModel):
        result: str

    class MixedModule(Module):
        model = "test-model"
        tools = [external_tool]
        final_output = Output

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

    class Output(BaseModel):
        result: str

    class ConflictModule(Module):
        model = "test-model"
        tools = [my_tool, my_tool_copy]
        final_output = Output

    with pytest.raises(ToolConflictError, match="Duplicate tool name"):
        ConflictModule()


def test_system_prompt_from_string():
    """Test system prompt from string."""
    class Output(BaseModel):
        result: str

    class StringPromptModule(Module):
        model = "test-model"
        system_prompt = "You are a helpful assistant."
        final_output = Output

    mod = StringPromptModule()
    msg = mod._build_system_message()
    assert msg["role"] == "system"
    assert msg["content"] == "You are a helpful assistant."


def test_system_prompt_empty():
    """Test empty system prompt."""
    class Output(BaseModel):
        result: str

    class NoPromptModule(Module):
        model = "test-model"
        final_output = Output

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
        model = "test-model"
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
        model = "test-model"
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
    class Output(BaseModel):
        result: str

    class SingleTurnModule(Module):
        model = "test-model"
        max_steps = None
        final_output = Output

    mod = SingleTurnModule()
    assert mod.max_steps is None


def test_init_validation_single_turn_requires_final_output():
    """Test that single-turn mode requires final_output."""
    class InvalidModule(Module):
        model = "test-model"
        final_output = None
        # max_steps = None (default)

    with pytest.raises(ValueError, match="final_output must be defined for single-turn"):
        InvalidModule()
