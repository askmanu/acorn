"""Tests for parse error retry logic."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from acorn import Module
from acorn.exceptions import ParseError


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


def test_parse_retry_success_on_second_attempt():
    """Test successful retry after initial parse error."""
    class Output(BaseModel):
        count: int

    class CountModule(Module):
        final_output = Output
        max_parse_retries = 2

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # First call: invalid data
        invalid_call = create_tool_call("__finish__", {"count": "not_a_number"})

        # Second call (retry): valid data
        valid_call = create_tool_call("__finish__", {"count": 42})

        # Set up mock to return different responses
        mock_completion.side_effect = [
            MockResponse(tool_calls=[invalid_call]),
            MockResponse(tool_calls=[valid_call])
        ]

        mod = CountModule()
        result = mod()

        assert result.count == 42
        # Should have been called twice (initial + 1 retry)
        assert mock_completion.call_count == 2


def test_parse_retry_failure_after_max_retries():
    """Test that ParseError is raised after max retries exhausted."""
    class Output(BaseModel):
        count: int

    class CountModule(Module):
        final_output = Output
        max_parse_retries = 2

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # All calls return invalid data
        invalid_call = create_tool_call("__finish__", {"count": "invalid"})
        mock_completion.return_value = MockResponse(tool_calls=[invalid_call])

        mod = CountModule()

        with pytest.raises(ParseError, match="Failed to validate output after 2 retries"):
            mod()

        # Should have been called 3 times (initial + 2 retries)
        assert mock_completion.call_count == 3


def test_parse_retry_includes_error_message():
    """Test that retry includes error details."""
    class Output(BaseModel):
        value: int

    class TestModule(Module):
        final_output = Output
        max_parse_retries = 1

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        invalid_call = create_tool_call("__finish__", {"value": "bad"})
        valid_call = create_tool_call("__finish__", {"value": 10})

        mock_completion.side_effect = [
            MockResponse(tool_calls=[invalid_call]),
            MockResponse(tool_calls=[valid_call])
        ]

        mod = TestModule()
        result = mod()

        # Check that retry message includes error
        second_call_args = mock_completion.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Should have error message
        error_msg = messages[-1]
        assert error_msg["role"] == "user"
        assert "Error" in error_msg["content"]
        assert "validation failed" in error_msg["content"]


def test_parse_retry_with_zero_retries():
    """Test that max_parse_retries=0 means no retries."""
    class Output(BaseModel):
        value: int

    class NoRetryModule(Module):
        final_output = Output
        max_parse_retries = 0

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        invalid_call = create_tool_call("__finish__", {"value": "bad"})
        mock_completion.return_value = MockResponse(tool_calls=[invalid_call])

        mod = NoRetryModule()

        with pytest.raises(ParseError, match="Failed to validate output after 0 retries"):
            mod()

        # Should only be called once
        assert mock_completion.call_count == 1


def test_parse_retry_success_on_first_attempt():
    """Test that no retry happens when first attempt succeeds."""
    class Output(BaseModel):
        value: int

    class SuccessModule(Module):
        final_output = Output
        max_parse_retries = 2

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        valid_call = create_tool_call("__finish__", {"value": 100})
        mock_completion.return_value = MockResponse(tool_calls=[valid_call])

        mod = SuccessModule()
        result = mod()

        assert result.value == 100
        # Should only be called once (no retries needed)
        assert mock_completion.call_count == 1


def test_system_prompt_from_path(tmp_path):
    """Test loading system prompt from file path."""
    # Create a temporary file with system prompt
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("You are a helpful assistant from a file.")

    class Output(BaseModel):
        result: str

    class FilePromptModule(Module):
        system_prompt = prompt_file
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        valid_call = create_tool_call("__finish__", {"result": "ok"})
        mock_completion.return_value = MockResponse(tool_calls=[valid_call])

        mod = FilePromptModule()
        result = mod()

        # Check system message
        call_args = mock_completion.call_args
        messages = call_args.kwargs["messages"]

        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a helpful assistant from a file."
