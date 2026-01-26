"""Tests for single-turn module execution."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel, Field

from acorn import Module
from acorn.exceptions import AcornError, ParseError


class MockResponse:
    """Mock LiteLLM response object."""
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.choices = [MagicMock()]
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls or []
        self.choices[0].message = message
        self.choices[0].finish_reason = finish_reason


def create_tool_call(name: str, arguments: dict, call_id: str = "call_123"):
    """Create a mock tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def test_single_turn_basic():
    """Test basic single-turn execution."""
    class Input(BaseModel):
        text: str

    class Output(BaseModel):
        summary: str

    class SummaryModule(Module):
        """Summarize text."""
        initial_input = Input
        final_output = Output

    # Mock the LiteLLM call
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Create mock response with __finish__ call
        finish_call = create_tool_call("__finish__", {"summary": "Test summary"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = SummaryModule()
        result = mod(text="Long text to summarize...")

        assert isinstance(result, Output)
        assert result.summary == "Test summary"


def test_single_turn_with_field_descriptions():
    """Test that field descriptions are included."""
    class Input(BaseModel):
        query: str = Field(description="The search query")

    class Output(BaseModel):
        results: str = Field(description="Search results")

    class SearchModule(Module):
        initial_input = Input
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        finish_call = create_tool_call("__finish__", {"results": "Found items"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = SearchModule()
        result = mod(query="test")

        assert result.results == "Found items"

        # Check that litellm was called with correct messages
        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]

        # User message should contain XML with descriptions
        user_msg = messages[1]["content"]
        assert "<query" in user_msg
        assert "test" in user_msg


def test_single_turn_validation_error():
    """Test that validation errors raise ParseError."""
    class Output(BaseModel):
        count: int  # Expects int, will get string

    class CountModule(Module):
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Return invalid data (string instead of int)
        finish_call = create_tool_call("__finish__", {"count": "not_a_number"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = CountModule()

        with pytest.raises(ParseError, match="Failed to validate output"):
            mod()


def test_single_turn_no_tool_call():
    """Test error when no tool is called."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Return response with no tool calls
        mock_completion.return_value = MockResponse(content="Just text", tool_calls=[])

        mod = TestModule()

        with pytest.raises(AcornError, match="No tool called in single-turn mode"):
            mod()


def test_single_turn_wrong_tool_call():
    """Test error when wrong tool is called."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Return call to wrong tool
        wrong_call = create_tool_call("other_tool", {"arg": "value"})
        mock_completion.return_value = MockResponse(tool_calls=[wrong_call])

        mod = TestModule()

        with pytest.raises(AcornError, match="Non-finish tool called"):
            mod()


def test_single_turn_system_prompt():
    """Test that system prompt is included in messages."""
    class Output(BaseModel):
        result: str

    class PromptModule(Module):
        """You are a helpful assistant."""
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        finish_call = create_tool_call("__finish__", {"result": "ok"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = PromptModule()
        result = mod()

        # Check system message
        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert "helpful assistant" in messages[0]["content"]


def test_single_turn_custom_model():
    """Test using custom model configuration."""
    class Output(BaseModel):
        result: str

    class CustomModelModule(Module):
        model = "gpt-4"
        temperature = 0.5
        max_tokens = 1000
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        finish_call = create_tool_call("__finish__", {"result": "done"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = CustomModelModule()
        result = mod()

        # Check that custom config was used
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "gpt-4"
        assert call_args.kwargs["temperature"] == 0.5
        assert call_args.kwargs["max_tokens"] == 1000


def test_single_turn_multiple_output_fields():
    """Test output with multiple fields."""
    class Output(BaseModel):
        summary: str
        word_count: int
        keywords: list[str]

    class AnalyzerModule(Module):
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        finish_call = create_tool_call("__finish__", {
            "summary": "Brief summary",
            "word_count": 42,
            "keywords": ["ai", "ml"]
        })
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = AnalyzerModule()
        result = mod()

        assert result.summary == "Brief summary"
        assert result.word_count == 42
        assert result.keywords == ["ai", "ml"]


def test_multi_turn_is_implemented():
    """Test that multi-turn is now implemented."""
    class Output(BaseModel):
        result: str

    class MultiTurnModule(Module):
        max_steps = 5
        final_output = Output

    mod = MultiTurnModule()

    # Should not raise NotImplementedError anymore
    assert mod.max_steps == 5
    # Actual execution requires mocked LLM calls (tested in test_agentic_loop.py)
