"""Tests for modules without final_output."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from acorn import Module, tool
from acorn.exceptions import AcornError


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


def test_single_turn_requires_final_output():
    """Single-turn mode must have final_output."""
    class NoOutputModule(Module):
        model = "test-model"
        max_steps = None  # Single-turn
        final_output = None

    with pytest.raises(ValueError, match="final_output must be defined for single-turn"):
        NoOutputModule()


def test_multi_turn_without_final_output_reaches_max_steps():
    """Multi-turn without final_output runs until max_steps."""
    @tool
    def action(msg: str) -> str:
        """Execute an action."""
        return f"Executed: {msg}"

    class ToolOnlyModule(Module):
        model = "test-model"
        max_steps = 2
        final_output = None
        tools = [action]

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: action("test1")
        # Step 2: action("test2")
        # Reaches max_steps → returns None

        action_call_1 = create_tool_call("action", {"msg": "test1"}, "call_1")
        action_call_2 = create_tool_call("action", {"msg": "test2"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[action_call_1]),
            MockResponse(tool_calls=[action_call_2]),
        ]

        mod = ToolOnlyModule()
        result = mod()

        assert result is None
        assert mock_completion.call_count == 2

        # Verify __finish__ WAS in tool schemas (but parameterless)
        for call in mock_completion.call_args_list:
            tools = call.kwargs.get("tools", [])
            tool_names = [t["function"]["name"] for t in tools]
            assert "__finish__" in tool_names

            # Find __finish__ and verify it has no required parameters
            finish_tool = next(t for t in tools if t["function"]["name"] == "__finish__")
            assert finish_tool["function"]["parameters"]["required"] == []


def test_multi_turn_without_final_output_early_finish():
    """Multi-turn without final_output can call __finish__ to end early."""
    @tool
    def action(msg: str) -> str:
        """Execute an action."""
        return f"Executed: {msg}"

    class ToolOnlyModule(Module):
        model = "test-model"
        max_steps = 5
        final_output = None
        tools = [action]

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: action("test1")
        # Step 2: __finish__() → returns None early

        action_call = create_tool_call("action", {"msg": "test1"}, "call_1")
        finish_call = create_tool_call("__finish__", {}, "call_2")  # No arguments

        mock_completion.side_effect = [
            MockResponse(tool_calls=[action_call]),
            MockResponse(tool_calls=[finish_call]),
        ]

        mod = ToolOnlyModule()
        result = mod()

        assert result is None
        assert mock_completion.call_count == 2  # Ended early (before max_steps=5)


def test_multi_turn_no_final_output_on_step_callback():
    """Test on_step callback works without final_output."""
    steps_collected = []

    @tool
    def log_data(data: str) -> str:
        """Log data."""
        return f"Logged: {data}"

    class CallbackModule(Module):
        model = "test-model"
        max_steps = 2
        final_output = None
        tools = [log_data]

        def on_step(self, step):
            steps_collected.append(step)
            return step

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        log_call = create_tool_call("log_data", {"data": "info"})
        mock_completion.return_value = MockResponse(tool_calls=[log_call])

        mod = CallbackModule()
        result = mod()

        assert result is None
        assert len(steps_collected) == 2
        assert all(s.counter > 0 for s in steps_collected)


def test_multi_turn_no_final_output_history_tracking():
    """Test history is tracked correctly without final_output."""
    @tool
    def get_info() -> str:
        """Get info."""
        return "Some info"

    class HistoryModule(Module):
        model = "test-model"
        max_steps = 1
        final_output = None
        tools = [get_info]

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        info_call = create_tool_call("get_info", {})
        mock_completion.return_value = MockResponse(tool_calls=[info_call])

        mod = HistoryModule()
        result = mod()

        assert result is None

        # History should contain: system, user (initial), assistant, tool result
        assert len(mod.history) >= 3
        assert mod.history[0]["role"] == "system"
        assert any(msg["role"] == "assistant" for msg in mod.history)
        assert any(msg["role"] == "tool" for msg in mod.history)


def test_multi_turn_no_final_output_step_finish():
    """Test step.finish() works without final_output."""
    @tool
    def check_condition() -> bool:
        """Check condition."""
        return True

    class EarlyExitModule(Module):
        model = "test-model"
        max_steps = 5
        final_output = None
        tools = [check_condition]

        def on_step(self, step):
            # Exit early after first tool call
            if step.counter == 1:
                step.finish()  # No kwargs needed
            return step

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        check_call = create_tool_call("check_condition", {})
        mock_completion.return_value = MockResponse(tool_calls=[check_call])

        mod = EarlyExitModule()
        result = mod()

        assert result is None
        assert mock_completion.call_count == 1  # Only one step before early exit


def test_multi_turn_allows_final_output_still_works():
    """Verify modules WITH final_output still work correctly."""
    @tool
    def process() -> str:
        """Process data."""
        return "processed"

    class Output(BaseModel):
        result: str

    class NormalModule(Module):
        model = "test-model"
        max_steps = 2
        final_output = Output
        tools = [process]

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        process_call = create_tool_call("process", {}, "call_1")
        finish_call = create_tool_call("__finish__", {"result": "done"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[process_call]),
            MockResponse(tool_calls=[finish_call]),
        ]

        mod = NormalModule()
        result = mod()

        assert result is not None
        assert isinstance(result, Output)
        assert result.result == "done"
