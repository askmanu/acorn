"""Tests for Phase 8: Partial Streaming for structured outputs."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from acorn import Module
from acorn.types import StreamChunk
from acorn.partial import Partial
from acorn.llm.litellm_client import _parse_partial_json


# Test Models
class SimpleOutput(BaseModel):
    name: str
    age: int


class ComplexOutput(BaseModel):
    title: str
    count: int
    summary: str
    tags: list[str]


# Mock Streaming Chunk Generator
def create_streaming_chunk_with_tool_delta(idx, tool_id, name=None, args_delta=""):
    """Create mock LiteLLM streaming chunk with tool call delta."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]

    delta = MagicMock()
    tc_delta = MagicMock()
    tc_delta.index = idx

    if name:
        tc_delta.id = tool_id
        tc_delta.function = MagicMock()
        tc_delta.function.name = name
        tc_delta.function.arguments = ""
    elif args_delta:
        tc_delta.id = None
        tc_delta.function = MagicMock()
        tc_delta.function.name = None
        tc_delta.function.arguments = args_delta
    else:
        tc_delta.id = None
        tc_delta.function = MagicMock()
        tc_delta.function.name = None
        tc_delta.function.arguments = ""

    delta.content = None
    delta.tool_calls = [tc_delta]
    chunk.choices[0].delta = delta
    chunk.choices[0].finish_reason = None

    return chunk


def create_final_streaming_chunk():
    """Create final streaming chunk with finish_reason."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = None
    chunk.choices[0].delta.tool_calls = []
    chunk.choices[0].finish_reason = "stop"
    return chunk


# Test Cases for _parse_partial_json


def test_parse_partial_json_simple_complete():
    """Test parsing complete JSON for simple model."""
    json_str = '{"name": "Alice", "age": 30}'
    result = _parse_partial_json(json_str, SimpleOutput)

    assert result is not None
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_partial_json_simple_incomplete():
    """Test parsing incomplete JSON for simple model."""
    json_str = '{"name": "Alice"'
    result = _parse_partial_json(json_str, SimpleOutput)

    assert result is not None
    assert result.name == "Alice"
    assert result.age is None


def test_parse_partial_json_empty():
    """Test parsing empty string returns None."""
    result = _parse_partial_json("", SimpleOutput)
    assert result is None

    result = _parse_partial_json("   ", SimpleOutput)
    assert result is None


def test_parse_partial_json_partial_field():
    """Test parsing JSON with partial field value."""
    json_str = '{"name": "Ali'
    result = _parse_partial_json(json_str, SimpleOutput)

    # Should return empty partial since field value is incomplete
    assert result is not None
    # name field might be None since parsing couldn't complete it
    assert result.age is None


def test_parse_partial_json_complex_complete():
    """Test parsing complete JSON for complex model."""
    json_str = '{"title": "Test", "count": 5, "summary": "A test", "tags": ["a", "b"]}'
    result = _parse_partial_json(json_str, ComplexOutput)

    assert result is not None
    assert result.title == "Test"
    assert result.count == 5
    assert result.summary == "A test"
    assert result.tags == ["a", "b"]


def test_parse_partial_json_complex_partial():
    """Test parsing partially complete JSON for complex model."""
    json_str = '{"title": "Test", "count": 5, "summary":'
    result = _parse_partial_json(json_str, ComplexOutput)

    assert result is not None
    assert result.title == "Test"
    assert result.count == 5
    assert result.summary is None
    assert result.tags is None


def test_parse_partial_json_with_comma():
    """Test parsing JSON ending with comma."""
    json_str = '{"name": "Alice", "age": 30,'
    result = _parse_partial_json(json_str, SimpleOutput)

    assert result is not None
    assert result.name == "Alice"
    assert result.age == 30


def test_parse_partial_json_invalid_returns_empty_partial():
    """Test that unparsable JSON returns empty partial."""
    json_str = 'totally invalid json{{{['
    result = _parse_partial_json(json_str, SimpleOutput)

    # Should return empty partial as fallback
    assert result is not None
    assert result.name is None
    assert result.age is None


# Integration Tests with Module Streaming


def test_streaming_finish_with_partial_updates():
    """Test that __finish__ calls generate partial updates during streaming."""

    collected_chunks = []

    class TestModule(Module):
        model = "test-model"
        stream = True
        final_output = SimpleOutput

        def on_stream(self, chunk):
            collected_chunks.append(chunk)

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Simulate streaming __finish__ arguments progressively
        def streaming_response():
            yield create_streaming_chunk_with_tool_delta(0, "call_1", "__finish__", "")
            yield create_streaming_chunk_with_tool_delta(0, None, None, '{"name"')
            yield create_streaming_chunk_with_tool_delta(0, None, None, ': "Alice"')
            yield create_streaming_chunk_with_tool_delta(0, None, None, ', "age": ')
            yield create_streaming_chunk_with_tool_delta(0, None, None, '30}')
            yield create_final_streaming_chunk()

        mock_completion.return_value = streaming_response()

        mod = TestModule()
        result = mod()

    # Verify partial instances were created
    partial_chunks = [c for c in collected_chunks if c.partial is not None]

    # Should have at least some partial updates
    assert len(partial_chunks) >= 1


def test_streaming_finish_progressive_fields():
    """Test that partial fields appear progressively as JSON streams."""

    collected_chunks = []

    class TestModule(Module):
        model = "test-model"
        stream = True
        max_steps = 3
        final_output = SimpleOutput

        def on_stream(self, chunk):
            collected_chunks.append(chunk)

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Simulate streaming __finish__ with progressive field completion
        def streaming_response():
            # Start __finish__ call
            yield create_streaming_chunk_with_tool_delta(0, "call_1", "__finish__", "")
            # Stream opening brace and first field
            yield create_streaming_chunk_with_tool_delta(0, None, None, '{"name": "Alice"')
            # Complete first field and start second
            yield create_streaming_chunk_with_tool_delta(0, None, None, ', "age": 30}')
            yield create_final_streaming_chunk()

        mock_completion.return_value = streaming_response()

        mod = TestModule()
        result = mod()

    # Verify we got partial chunks
    partial_chunks = [c for c in collected_chunks if c.partial is not None]
    assert len(partial_chunks) >= 1

    # Last partial should have both fields (or at least name)
    last_partial = partial_chunks[-1].partial
    assert last_partial.name == "Alice"


def test_streaming_non_finish_tool_no_partial():
    """Test that non-__finish__ tools don't generate partial updates."""

    collected_chunks = []

    class TestModule(Module):
        model = "test-model"
        stream = True
        max_steps = 3
        final_output = SimpleOutput

        def on_stream(self, chunk):
            collected_chunks.append(chunk)

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Simulate streaming a different tool (not __finish__)
        def streaming_response():
            yield create_streaming_chunk_with_tool_delta(0, "call_1", "some_other_tool", "")
            yield create_streaming_chunk_with_tool_delta(0, None, None, '{"param": "value"}')
            yield create_final_streaming_chunk()

        mock_completion.return_value = streaming_response()

        mod = TestModule()
        # This should timeout or fail since no __finish__ is called
        # For this test, we just verify no partial chunks were generated
        try:
            result = mod()
        except:
            pass  # Expected to fail since no __finish__

    # Verify NO partial chunks were generated (only tool_call chunks)
    partial_chunks = [c for c in collected_chunks if c.partial is not None]
    assert len(partial_chunks) == 0


def test_streaming_with_no_schema():
    """Test that streaming without final_output schema doesn't break."""

    collected_chunks = []

    class TestModuleNoSchema(Module):
        model = "test-model"
        stream = True
        max_steps = 3
        # No final_output defined

        def on_stream(self, chunk):
            collected_chunks.append(chunk)

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        def streaming_response():
            yield create_streaming_chunk_with_tool_delta(0, "call_1", "__finish__", "")
            yield create_streaming_chunk_with_tool_delta(0, None, None, '{"result": "test"}')
            yield create_final_streaming_chunk()

        mock_completion.return_value = streaming_response()

        mod = TestModuleNoSchema()
        try:
            result = mod()
        except:
            pass  # Expected to fail for other reasons

    # Should not have partial chunks since no schema
    partial_chunks = [c for c in collected_chunks if c.partial is not None]
    assert len(partial_chunks) == 0


def test_partial_streaming_complex_types():
    """Test partial streaming with complex nested types."""

    collected_chunks = []

    class TestModuleComplex(Module):
        model = "test-model"
        stream = True
        final_output = ComplexOutput

        def on_stream(self, chunk):
            collected_chunks.append(chunk)

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        def streaming_response():
            yield create_streaming_chunk_with_tool_delta(0, "call_1", "__finish__", "")
            yield create_streaming_chunk_with_tool_delta(0, None, None, '{"title": "Test"')
            yield create_streaming_chunk_with_tool_delta(0, None, None, ', "count": 5')
            yield create_streaming_chunk_with_tool_delta(0, None, None, ', "summary": "A summary"')
            yield create_streaming_chunk_with_tool_delta(0, None, None, ', "tags": ["a", "b"]}')
            yield create_final_streaming_chunk()

        mock_completion.return_value = streaming_response()

        mod = TestModuleComplex()
        result = mod()

    # Verify progressive partial updates
    partial_chunks = [c for c in collected_chunks if c.partial is not None]
    assert len(partial_chunks) >= 1

    # Check final partial has all fields
    if len(partial_chunks) > 0:
        last_partial = partial_chunks[-1].partial
        assert last_partial.title == "Test"
        assert last_partial.count == 5


def test_streaming_done_chunk_sent():
    """Test that final done=True chunk is sent after streaming completes."""

    collected_chunks = []

    class TestModule(Module):
        stream = True
        final_output = SimpleOutput

        def on_stream(self, chunk):
            collected_chunks.append(chunk)

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        def streaming_response():
            yield create_streaming_chunk_with_tool_delta(0, "call_1", "__finish__", "")
            yield create_streaming_chunk_with_tool_delta(0, None, None, '{"name": "Alice", "age": 30}')
            yield create_final_streaming_chunk()

        mock_completion.return_value = streaming_response()

        mod = TestModule()
        result = mod()

    # Verify final done chunk was sent
    done_chunks = [c for c in collected_chunks if c.done]
    assert len(done_chunks) >= 1
