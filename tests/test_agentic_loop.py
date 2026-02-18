"""Tests for multi-turn agentic loop execution."""

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


def test_agentic_loop_basic():
    """Test basic multi-turn execution."""
    call_count = 0

    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    class Output(BaseModel):
        answer: str

    class AgenticModule(Module):
        model = "test-model"
        max_steps = 3
        tools = [search]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: Call search tool
        search_call = create_tool_call("search", {"query": "test"}, "call_1")

        # Step 2: Call __finish__
        finish_call = create_tool_call("__finish__", {"answer": "Final answer"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[search_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = AgenticModule()
        result = mod()

        assert result.answer == "Final answer"
        assert mock_completion.call_count == 2


def test_agentic_loop_multiple_tools():
    """Test loop with multiple tool calls."""
    @tool
    def calculate(x: int, y: int) -> int:
        """Add two numbers."""
        return x + y

    @tool
    def format_result(value: int) -> str:
        """Format a result."""
        return f"Result is {value}"

    class Output(BaseModel):
        formatted: str

    class CalculatorModule(Module):
        model = "test-model"
        max_steps = 5
        tools = [calculate, format_result]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: Calculate
        calc_call = create_tool_call("calculate", {"x": 5, "y": 3}, "call_1")

        # Step 2: Format
        format_call = create_tool_call("format_result", {"value": 8}, "call_2")

        # Step 3: Finish
        finish_call = create_tool_call("__finish__", {"formatted": "Result is 8"}, "call_3")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[calc_call]),
            MockResponse(tool_calls=[format_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = CalculatorModule()
        result = mod()

        assert result.formatted == "Result is 8"
        assert mock_completion.call_count == 3


def test_agentic_loop_tool_error():
    """Test handling of tool execution errors."""
    @tool
    def failing_tool() -> str:
        """This tool will fail."""
        raise ValueError("Tool error")

    class Output(BaseModel):
        result: str

    class ErrorModule(Module):
        model = "test-model"
        max_steps = 3
        tools = [failing_tool]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: Call failing tool
        fail_call = create_tool_call("failing_tool", {}, "call_1")

        # Step 2: Call __finish__ after error
        finish_call = create_tool_call("__finish__", {"result": "recovered"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[fail_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = ErrorModule()
        result = mod()

        assert result.result == "recovered"

        # Check that error was added to history
        # The second call should have the error in messages
        second_call_args = mock_completion.call_args_list[1]
        messages = second_call_args.kwargs["messages"]

        # Find tool result message
        tool_messages = [m for m in messages if m.get("role") == "tool"]
        assert len(tool_messages) > 0
        assert "Error" in tool_messages[0]["content"]
        assert "Tool error" in tool_messages[0]["content"]


def test_agentic_loop_max_steps_forced_termination_tool_choice():
    """Test forced termination at max_steps using tool_choice."""
    @tool
    def endless_tool() -> str:
        """A tool that keeps running."""
        return "continue"

    class Output(BaseModel):
        result: str

    class EndlessModule(Module):
        model = "test-model"
        max_steps = 2
        tools = [endless_tool]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # First 2 steps: call endless_tool
        endless_call = create_tool_call("endless_tool", {}, "call_1")

        # Final step (forced): call __finish__
        finish_call = create_tool_call("__finish__", {"result": "forced output"}, "call_finish")

        # Mock responses: first 2 are endless, 3rd is forced __finish__
        mock_completion.side_effect = [
            MockResponse(tool_calls=[endless_call]),
            MockResponse(tool_calls=[endless_call]),
            MockResponse(tool_calls=[finish_call])  # Forced termination
        ]

        mod = EndlessModule()
        result = mod()

        # Should have forced termination
        assert result.result == "forced output"
        # Called max_steps + 1 (for forced termination)
        assert mock_completion.call_count == 3

        # Check that tool_choice was used in the last call
        last_call_kwargs = mock_completion.call_args_list[2].kwargs
        assert "tool_choice" in last_call_kwargs
        assert last_call_kwargs["tool_choice"]["function"]["name"] == "__finish__"


def test_agentic_loop_max_steps_forced_termination_raises_error():
    """Test forced termination raises AcornError when tool_choice fails."""
    @tool
    def endless_tool() -> str:
        """A tool that keeps running."""
        return "continue"

    class Output(BaseModel):
        result: str

    class EndlessModule(Module):
        model = "test-model"
        max_steps = 2
        tools = [endless_tool]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # First 2 steps: call endless_tool
        endless_call = create_tool_call("endless_tool", {}, "call_1")

        # Mock responses
        def mock_side_effect(*args, **kwargs):
            # If tool_choice is present, simulate tool_choice not supported
            if "tool_choice" in kwargs:
                raise Exception("tool_choice not supported")

            # Normal tool call
            return MockResponse(tool_calls=[endless_call])

        mock_completion.side_effect = mock_side_effect

        mod = EndlessModule()

        with pytest.raises(AcornError, match="reached max_steps"):
            mod()


def test_agentic_loop_history_accumulates():
    """Test that history accumulates correctly."""
    @tool
    def get_data() -> str:
        """Get data."""
        return "data"

    class Output(BaseModel):
        result: str

    class HistoryModule(Module):
        model = "test-model"
        max_steps = 3
        tools = [get_data]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        data_call = create_tool_call("get_data", {}, "call_1")
        finish_call = create_tool_call("__finish__", {"result": "done"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[data_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = HistoryModule()
        result = mod()

        # Check history grew
        assert len(mod.history) > 2  # system + user + assistant + tool result + assistant

        # Verify structure
        assert mod.history[0]["role"] == "system"
        assert mod.history[1]["role"] == "user"


def test_agentic_loop_on_step_callback():
    """Test on_step callback is called."""
    step_count = 0

    @tool
    def my_tool() -> str:
        """Test tool."""
        return "result"

    class Output(BaseModel):
        result: str

    class CallbackModule(Module):
        model = "test-model"
        max_steps = 3
        tools = [my_tool]
        final_output = Output

        def on_step(self, step):
            nonlocal step_count
            step_count += 1
            assert step.counter == step_count
            assert step.model == self.model
            return step

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        tool_call = create_tool_call("my_tool", {}, "call_1")
        finish_call = create_tool_call("__finish__", {"result": "done"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[tool_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = CallbackModule()
        result = mod()

        # on_step should have been called once (not for __finish__ step)
        assert step_count == 1


def test_agentic_loop_step_finish():
    """Test step.finish() terminates loop early."""
    @tool
    def my_tool() -> str:
        """Test tool."""
        return "data"

    class Output(BaseModel):
        result: str

    class EarlyExitModule(Module):
        model = "test-model"
        max_steps = 5
        tools = [my_tool]
        final_output = Output

        def on_step(self, step):
            # Exit after first step
            if step.counter == 1:
                step.finish(result="early exit")
            return step

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        tool_call = create_tool_call("my_tool", {}, "call_1")
        mock_completion.return_value = MockResponse(tool_calls=[tool_call])

        mod = EarlyExitModule()
        result = mod()

        assert result.result == "early exit"
        # Should only call LLM once
        assert mock_completion.call_count == 1


def test_agentic_loop_add_tool():
    """Test adding tools dynamically via on_step."""
    @tool
    def initial_tool() -> str:
        """Initial tool."""
        return "initial"

    @tool
    def added_tool() -> str:
        """Tool added during execution."""
        return "added"

    class Output(BaseModel):
        result: str

    class DynamicToolModule(Module):
        model = "test-model"
        max_steps = 5
        tools = [initial_tool]
        final_output = Output

        def on_step(self, step):
            if step.counter == 1:
                # Add new tool after first step
                step.add_tool(added_tool)
            return step

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: Use initial tool
        initial_call = create_tool_call("initial_tool", {}, "call_1")

        # Step 2: Use added tool
        added_call = create_tool_call("added_tool", {}, "call_2")

        # Step 3: Finish
        finish_call = create_tool_call("__finish__", {"result": "done"}, "call_3")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[initial_call]),
            MockResponse(tool_calls=[added_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = DynamicToolModule()
        result = mod()

        assert result.result == "done"


def test_agentic_loop_remove_tool():
    """Test removing tools dynamically via on_step."""
    @tool
    def tool_to_remove() -> str:
        """This tool will be removed."""
        return "temp"

    @tool
    def permanent_tool() -> str:
        """This tool stays."""
        return "perm"

    class Output(BaseModel):
        result: str

    class RemoveToolModule(Module):
        model = "test-model"
        max_steps = 5
        tools = [tool_to_remove, permanent_tool]
        final_output = Output

        def on_step(self, step):
            if step.counter == 1:
                step.remove_tool("tool_to_remove")
            return step

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: Use tool that will be removed
        temp_call = create_tool_call("tool_to_remove", {}, "call_1")

        # Step 2: Try to use removed tool (should fail)
        # Step 2: Use permanent tool instead
        perm_call = create_tool_call("permanent_tool", {}, "call_2")

        # Step 3: Finish
        finish_call = create_tool_call("__finish__", {"result": "done"}, "call_3")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[temp_call]),
            MockResponse(tool_calls=[perm_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = RemoveToolModule()
        result = mod()

        assert result.result == "done"


def test_agentic_loop_retries_on_no_tool_calls():
    """Test that agentic loop retries when model returns no tool calls."""
    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    class Output(BaseModel):
        answer: str

    class RetryModule(Module):
        model = "test-model"
        max_steps = 5
        tools = [search]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Step 1: No tool calls (text only)
        # Step 2 (retry): Calls search tool
        # Step 3: Calls __finish__
        search_call = create_tool_call("search", {"query": "test"}, "call_1")
        finish_call = create_tool_call("__finish__", {"answer": "done"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(content="Let me think about this..."),
            MockResponse(tool_calls=[search_call]),
            MockResponse(tool_calls=[finish_call])
        ]

        mod = RetryModule()
        result = mod()

        assert result.answer == "done"
        # Verify reminder message was appended to history
        reminder_msgs = [
            m for m in mod.history
            if m.get("role") == "user" and "must respond by calling" in m.get("content", "")
        ]
        assert len(reminder_msgs) == 1


def test_agentic_loop_fails_after_max_retries_no_tool_calls():
    """Test that agentic loop raises error after max_parse_retries with no tool calls."""
    @tool
    def search(query: str) -> str:
        """Search for information."""
        return f"Results for: {query}"

    class Output(BaseModel):
        answer: str

    class RetryFailModule(Module):
        model = "test-model"
        max_steps = 10
        max_parse_retries = 2
        tools = [search]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Always return no tool calls
        mock_completion.return_value = MockResponse(content="Just thinking...")

        mod = RetryFailModule()

        with pytest.raises(AcornError, match="No tool calls in agentic loop step after 2 retries"):
            mod()
