"""Integration tests for cache configuration."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from acorn import Module


class Input(BaseModel):
    text: str


class Output(BaseModel):
    result: str


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


def test_single_turn_with_cache_true():
    """Test single-turn module with cache=True."""
    class CachedModule(Module):
        """Test module."""
        initial_input = Input
        final_output = Output
        cache = True

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Mock the __finish__ call
        finish_call = create_tool_call("__finish__", {"result": "test output"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        module = CachedModule()
        result = module(text="test input")

        # Verify cache was passed to litellm
        call_args = mock_completion.call_args
        assert call_args.kwargs["cache_control_injection_points"] == [
            {"location": "message", "role": "system"},
            {"location": "message", "index": 0}
        ]

        # Verify output
        assert result.result == "test output"


def test_single_turn_with_custom_cache():
    """Test single-turn module with custom cache configuration."""
    class CustomCachedModule(Module):
        """Test module."""
        initial_input = Input
        final_output = Output
        cache = [{"location": "message", "role": "system"}]

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Mock the __finish__ call
        finish_call = create_tool_call("__finish__", {"result": "custom cache output"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        module = CustomCachedModule()
        result = module(text="test input")

        # Verify custom cache was passed to litellm
        call_args = mock_completion.call_args
        assert call_args.kwargs["cache_control_injection_points"] == [
            {"location": "message", "role": "system"}
        ]

        # Verify output
        assert result.result == "custom cache output"


def test_module_without_cache():
    """Test module without cache (backwards compatibility)."""
    class NoCacheModule(Module):
        """Test module."""
        initial_input = Input
        final_output = Output
        # cache not specified (defaults to None)

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Mock the __finish__ call
        finish_call = create_tool_call("__finish__", {"result": "no cache output"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        module = NoCacheModule()
        result = module(text="test input")

        # Verify cache_control_injection_points was NOT added
        call_args = mock_completion.call_args
        assert "cache_control_injection_points" not in call_args.kwargs

        # Verify output
        assert result.result == "no cache output"


def test_multi_turn_with_cache():
    """Test multi-turn agentic mode with cache."""

    def mock_tool(value: str) -> str:
        """A test tool."""
        return f"processed: {value}"

    class MultiTurnCached(Module):
        """Multi-turn agent with cache."""
        max_steps = 2
        tools = [mock_tool]
        final_output = Output
        cache = True

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: Call mock_tool
        # Step 2: Call __finish__
        tool_call = create_tool_call("mock_tool", {"value": "test"}, "call_1")
        finish_call = create_tool_call("__finish__", {"result": "final result"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(content="Using tool", tool_calls=[tool_call]),
            MockResponse(content="Done", tool_calls=[finish_call])
        ]

        module = MultiTurnCached()
        result = module()

        # Verify cache was passed to both calls
        assert mock_completion.call_count == 2
        for call in mock_completion.call_args_list:
            assert call.kwargs["cache_control_injection_points"] == [
                {"location": "message", "role": "system"},
                {"location": "message", "index": 0}
            ]

        # Verify output
        assert result.result == "final result"
