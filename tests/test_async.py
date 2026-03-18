"""Tests for async-specific functionality."""

import asyncio
import json
import time
from unittest.mock import patch, AsyncMock, MagicMock
from pydantic import BaseModel

from acorn import Module, tool


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


class Output(BaseModel):
    result: str


# =============================================================================
# Test: Async tool execution
# =============================================================================

async def test_async_tool_execution():
    """Test that async tools work correctly."""
    @tool
    async def async_search(query: str) -> str:
        """Search asynchronously."""
        await asyncio.sleep(0)  # Simulate async work
        return f"Async results for: {query}"

    class AgentModule(Module):
        model = "test-model"
        max_steps = 3
        tools = [async_search]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.acompletion', new_callable=AsyncMock) as mock_completion:
        search_call = create_tool_call("async_search", {"query": "test"}, "call_1")
        finish_call = create_tool_call("__finish__", {"result": "done"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[search_call]),
            MockResponse(tool_calls=[finish_call]),
        ]

        mod = AgentModule()
        result = await mod()

        assert result.result == "done"

        # Verify tool result was added to history
        tool_msgs = [m for m in mod.history if m.get("role") == "tool"]
        assert any("Async results for: test" in m["content"] for m in tool_msgs)


# =============================================================================
# Test: Parallel tool execution
# =============================================================================

async def test_parallel_tool_execution():
    """Test that multiple tool calls in one step run concurrently."""
    execution_log = []

    @tool
    async def slow_tool_a(value: str) -> str:
        """Slow tool A."""
        execution_log.append(("a_start", time.monotonic()))
        await asyncio.sleep(0.1)
        execution_log.append(("a_end", time.monotonic()))
        return f"A: {value}"

    @tool
    async def slow_tool_b(value: str) -> str:
        """Slow tool B."""
        execution_log.append(("b_start", time.monotonic()))
        await asyncio.sleep(0.1)
        execution_log.append(("b_end", time.monotonic()))
        return f"B: {value}"

    class ParallelModule(Module):
        model = "test-model"
        max_steps = 3
        tools = [slow_tool_a, slow_tool_b]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.acompletion', new_callable=AsyncMock) as mock_completion:
        # Step 1: Call both tools in parallel
        call_a = create_tool_call("slow_tool_a", {"value": "x"}, "call_a")
        call_b = create_tool_call("slow_tool_b", {"value": "y"}, "call_b")

        # Step 2: Finish
        finish_call = create_tool_call("__finish__", {"result": "parallel done"}, "call_f")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[call_a, call_b]),
            MockResponse(tool_calls=[finish_call]),
        ]

        mod = ParallelModule()
        start = time.monotonic()
        result = await mod()
        elapsed = time.monotonic() - start

        assert result.result == "parallel done"

        # Both tools should have started before either finished (parallel execution)
        starts = [t for label, t in execution_log if label.endswith("_start")]
        ends = [t for label, t in execution_log if label.endswith("_end")]

        assert len(starts) == 2
        assert len(ends) == 2

        # The second tool should start before the first tool ends
        # (proving concurrent execution via asyncio.gather)
        assert starts[1] < ends[0], "Tools should run concurrently, not sequentially"


# =============================================================================
# Test: Mixed sync/async tools
# =============================================================================

async def test_mixed_sync_async_tools():
    """Test that sync and async tools can coexist in the same module."""
    @tool
    def sync_tool(x: int) -> int:
        """Sync tool."""
        return x * 2

    @tool
    async def async_tool(x: int) -> int:
        """Async tool."""
        await asyncio.sleep(0)
        return x * 3

    class MixedModule(Module):
        model = "test-model"
        max_steps = 5
        tools = [sync_tool, async_tool]
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.acompletion', new_callable=AsyncMock) as mock_completion:
        sync_call = create_tool_call("sync_tool", {"x": 5}, "call_1")
        async_call = create_tool_call("async_tool", {"x": 5}, "call_2")
        finish_call = create_tool_call("__finish__", {"result": "mixed done"}, "call_3")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[sync_call, async_call]),
            MockResponse(tool_calls=[finish_call]),
        ]

        mod = MixedModule()
        result = await mod()

        assert result.result == "mixed done"

        # Verify both tool results in history
        tool_msgs = [m for m in mod.history if m.get("role") == "tool"]
        assert any("10" in m["content"] for m in tool_msgs)  # sync: 5*2
        assert any("15" in m["content"] for m in tool_msgs)  # async: 5*3


# =============================================================================
# Test: Async on_step callback
# =============================================================================

async def test_async_on_step_callback():
    """Test that async on_step callback works."""
    step_data = []

    @tool
    def my_tool() -> str:
        """Test tool."""
        return "result"

    class AsyncCallbackModule(Module):
        model = "test-model"
        max_steps = 3
        tools = [my_tool]
        final_output = Output

        async def on_step(self, step):
            await asyncio.sleep(0)  # Simulate async work
            step_data.append(step.counter)
            return step

    with patch('acorn.llm.litellm_client.litellm.acompletion', new_callable=AsyncMock) as mock_completion:
        tool_call = create_tool_call("my_tool", {}, "call_1")
        finish_call = create_tool_call("__finish__", {"result": "done"}, "call_2")

        mock_completion.side_effect = [
            MockResponse(tool_calls=[tool_call]),
            MockResponse(tool_calls=[finish_call]),
        ]

        mod = AsyncCallbackModule()
        result = await mod()

        assert result.result == "done"
        assert step_data == [1]


# =============================================================================
# Test: Async on_stream callback
# =============================================================================

async def test_async_on_stream_callback():
    """Test that async on_stream callback works."""
    from acorn.llm.litellm_client import _handle_streaming_response
    from acorn.types import StreamChunk

    collected = []

    async def async_callback(chunk):
        await asyncio.sleep(0)
        collected.append(chunk)

    async def mock_stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        delta = MagicMock()
        delta.content = "hello"
        delta.tool_calls = []
        chunk.choices[0].delta = delta
        chunk.choices[0].finish_reason = None
        yield chunk

        # Final chunk
        final = MagicMock()
        final.choices = [MagicMock()]
        final.choices[0].delta = MagicMock()
        final.choices[0].delta.content = None
        final.choices[0].delta.tool_calls = []
        final.choices[0].finish_reason = "stop"
        yield final

    result = await _handle_streaming_response(mock_stream(), async_callback)

    assert result["content"] == "hello"
    # Should have content chunk + done chunk
    assert len(collected) >= 2
    assert collected[-1].done is True


# =============================================================================
# Test: run() sync wrapper
# =============================================================================

def test_run_sync_wrapper():
    """Test that run() provides sync access to async __call__."""
    class SyncModule(Module):
        model = "test-model"
        final_output = Output

    with patch('acorn.llm.litellm_client.litellm.acompletion', new_callable=AsyncMock) as mock_completion:
        finish_call = create_tool_call("__finish__", {"result": "sync result"})
        mock_completion.return_value = MockResponse(tool_calls=[finish_call])

        mod = SyncModule()
        # Use run() instead of await __call__()
        result = mod.run()

        assert isinstance(result, Output)
        assert result.result == "sync result"
