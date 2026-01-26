"""Tests for core data types."""

import pytest
from acorn.types import ToolCall, ToolResult, StreamChunk, Step


def test_tool_call_creation():
    """Test ToolCall instantiation."""
    tc = ToolCall(
        id="call_123",
        name="search",
        arguments={"query": "test"}
    )
    assert tc.id == "call_123"
    assert tc.name == "search"
    assert tc.arguments == {"query": "test"}


def test_tool_result_success():
    """Test ToolResult for successful execution."""
    tr = ToolResult(
        id="call_123",
        name="search",
        output=["result1", "result2"],
        error=None
    )
    assert tr.id == "call_123"
    assert tr.name == "search"
    assert tr.output == ["result1", "result2"]
    assert tr.error is None


def test_tool_result_error():
    """Test ToolResult for failed execution."""
    tr = ToolResult(
        id="call_123",
        name="search",
        output=None,
        error="Connection timeout"
    )
    assert tr.id == "call_123"
    assert tr.error == "Connection timeout"


def test_stream_chunk_content():
    """Test StreamChunk with text content."""
    chunk = StreamChunk(content="Hello", done=False)
    assert chunk.content == "Hello"
    assert chunk.partial is None
    assert chunk.tool_call is None
    assert chunk.done is False


def test_stream_chunk_partial():
    """Test StreamChunk with partial output."""
    partial_obj = {"name": "Alice"}
    chunk = StreamChunk(partial=partial_obj, done=False)
    assert chunk.partial == partial_obj
    assert chunk.content is None


def test_stream_chunk_tool_call():
    """Test StreamChunk with tool call."""
    tool_call = {"name": "search", "arguments": "{'q':"}
    chunk = StreamChunk(tool_call=tool_call, done=False)
    assert chunk.tool_call == tool_call


def test_stream_chunk_done():
    """Test StreamChunk with done flag."""
    chunk = StreamChunk(done=True)
    assert chunk.done is True


def test_step_creation():
    """Test Step instantiation."""
    tc = ToolCall(id="1", name="search", arguments={})
    tr = ToolResult(id="1", name="search", output="result")

    step = Step(
        counter=1,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        tools=["search"],
        response={"content": "test"},
        tool_calls=[tc],
        tool_results=[tr],
    )

    assert step.counter == 1
    assert step.model == "gpt-4"
    assert step.temperature == 0.7
    assert step.max_tokens == 1000
    assert len(step.tool_calls) == 1
    assert len(step.tool_results) == 1
    assert step._finished is False


def test_step_add_tool():
    """Test adding tools to a step."""
    step = Step(
        counter=1,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        tools=[],
        response={},
        tool_calls=[],
        tool_results=[],
    )

    def my_tool():
        pass

    step.add_tool(my_tool)
    assert len(step._tools_to_add) == 1
    assert step._tools_to_add[0] is my_tool


def test_step_remove_tool():
    """Test removing tools from a step."""
    step = Step(
        counter=1,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        tools=["search"],
        response={},
        tool_calls=[],
        tool_results=[],
    )

    step.remove_tool("search")
    assert "search" in step._tools_to_remove


def test_step_finish():
    """Test finishing a step."""
    step = Step(
        counter=1,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        tools=[],
        response={},
        tool_calls=[],
        tool_results=[],
    )

    assert step._finished is False
    step.finish(result="success", count=42)
    assert step._finished is True
    assert step._finish_kwargs == {"result": "success", "count": 42}


def test_step_mutations():
    """Test that step supports multiple mutations."""
    step = Step(
        counter=1,
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000,
        tools=["tool1"],
        response={},
        tool_calls=[],
        tool_results=[],
    )

    def new_tool():
        pass

    step.add_tool(new_tool)
    step.remove_tool("tool1")
    step.finish(output="done")

    assert len(step._tools_to_add) == 1
    assert len(step._tools_to_remove) == 1
    assert step._finished is True
