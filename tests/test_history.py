"""Tests for history tracking in modules."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from acorn import Module


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


def test_history_populated_single_turn():
    """Test that history is populated in single-turn mode."""
    class Input(BaseModel):
        text: str

    class Output(BaseModel):
        summary: str

    class TestModule(Module):
        initial_input = Input
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        finish_call = create_tool_call("__finish__", {"summary": "Test summary"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = TestModule()
        result = mod(text="Sample text")

        # History should have at least 2 messages: system, user, assistant
        assert len(mod.history) >= 2, "History should be populated"

        # Check structure
        assert mod.history[0]["role"] == "system"
        assert mod.history[1]["role"] == "user"
        assert "<text>" in mod.history[1]["content"]  # XML input

        # Should have assistant message
        assert mod.history[2]["role"] == "assistant"
        assert "tool_calls" in mod.history[2]


def test_history_accessible_after_execution():
    """Test that history can be accessed after module execution."""
    class Output(BaseModel):
        result: str

    class HistoryModule(Module):
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        finish_call = create_tool_call("__finish__", {"result": "done"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = HistoryModule()
        result = mod()

        # Should be able to iterate over history
        for msg in mod.history:
            assert "role" in msg
            assert msg["role"] in ["system", "user", "assistant", "tool"]


def test_history_multiple_executions():
    """Test that history resets between executions."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        finish_call = create_tool_call("__finish__", {"result": "done"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = TestModule()

        # First execution
        result1 = mod()
        history1_len = len(mod.history)

        # Second execution
        result2 = mod()
        history2_len = len(mod.history)

        # History should be reset/replaced, not accumulated
        assert history2_len == history1_len
