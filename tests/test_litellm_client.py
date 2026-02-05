"""Tests for LiteLLM client configuration."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from acorn.llm.litellm_client import call_llm, _handle_streaming_response, _parse_partial_json
from acorn.types import StreamChunk


def test_call_llm_with_string_model():
    """Test call_llm with string model."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4"
        )

        # Verify model was passed correctly
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "gpt-4"


def test_call_llm_with_dict_model_basic():
    """Test call_llm with dict model containing only id."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model={"id": "anthropic/claude-3-5-sonnet-20241022"}
        )

        # Verify model was extracted from id
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "anthropic/claude-3-5-sonnet-20241022"


def test_call_llm_with_vertex_parameters():
    """Test call_llm with Vertex AI parameters."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model={
                "id": "vertex_ai/gemini-pro",
                "vertex_location": "us-central1",
                "vertex_credentials": "path/to/creds.json"
            }
        )

        # Verify all parameters were passed
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "vertex_ai/gemini-pro"
        assert call_args.kwargs["vertex_location"] == "us-central1"
        assert call_args.kwargs["vertex_credentials"] == "path/to/creds.json"


def test_call_llm_with_reasoning_true():
    """Test call_llm with reasoning=True (should map to medium)."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model={"id": "gpt-4", "reasoning": True}
        )

        # Verify reasoning_effort is set to medium
        call_args = mock_completion.call_args
        assert call_args.kwargs["reasoning_effort"] == "medium"


def test_call_llm_with_reasoning_low():
    """Test call_llm with reasoning='low'."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model={"id": "gpt-4", "reasoning": "low"}
        )

        # Verify reasoning_effort is set to low
        call_args = mock_completion.call_args
        assert call_args.kwargs["reasoning_effort"] == "low"


def test_call_llm_with_reasoning_high():
    """Test call_llm with reasoning='high'."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model={"id": "gpt-4", "reasoning": "high"}
        )

        # Verify reasoning_effort is set to high
        call_args = mock_completion.call_args
        assert call_args.kwargs["reasoning_effort"] == "high"


def test_call_llm_with_all_parameters():
    """Test call_llm with all model parameters."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model={
                "id": "vertex_ai/gemini-pro",
                "vertex_location": "us-central1",
                "vertex_credentials": "creds.json",
                "reasoning": "medium"
            }
        )

        # Verify all parameters were passed correctly
        call_args = mock_completion.call_args
        assert call_args.kwargs["model"] == "vertex_ai/gemini-pro"
        assert call_args.kwargs["vertex_location"] == "us-central1"
        assert call_args.kwargs["vertex_credentials"] == "creds.json"
        assert call_args.kwargs["reasoning_effort"] == "medium"


# Test Models for Streaming
class TestOutput(BaseModel):
    result: str
    count: int


def create_tool_delta_chunk(idx, tool_id, name=None, args_delta=""):
    """Helper to create streaming chunk with tool delta."""
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


def create_finish_chunk():
    """Helper to create final streaming chunk."""
    chunk = MagicMock()
    chunk.choices = [MagicMock()]
    chunk.choices[0].delta = MagicMock()
    chunk.choices[0].delta.content = None
    chunk.choices[0].delta.tool_calls = []
    chunk.choices[0].finish_reason = "stop"
    return chunk


def test_handle_streaming_response_with_partial():
    """Test _handle_streaming_response creates Partial instances for __finish__."""
    collected_chunks = []
    
    def callback(chunk):
        collected_chunks.append(chunk)
    
    # Mock streaming response with __finish__ tool call
    def mock_stream():
        yield create_tool_delta_chunk(0, "call_1", "__finish__", "")
        yield create_tool_delta_chunk(0, None, None, '{"result": "test", "count": 5}')
        yield create_finish_chunk()
    
    result = _handle_streaming_response(
        mock_stream(),
        callback,
        final_output_schema=TestOutput
    )
    
    # Verify partial chunks were sent
    partial_chunks = [c for c in collected_chunks if c.partial]
    assert len(partial_chunks) >= 1
    assert partial_chunks[0].partial.result == "test"
    assert partial_chunks[0].partial.count == 5


def test_handle_streaming_response_without_schema():
    """Test streaming without schema sends tool_call deltas instead of partial."""
    collected_chunks = []
    
    def callback(chunk):
        collected_chunks.append(chunk)
    
    def mock_stream():
        yield create_tool_delta_chunk(0, "call_1", "__finish__", "")
        yield create_tool_delta_chunk(0, None, None, '{"result": "test"}')
        yield create_finish_chunk()
    
    result = _handle_streaming_response(
        mock_stream(),
        callback,
        final_output_schema=None
    )
    
    # Should NOT have partial chunks (no schema provided)
    partial_chunks = [c for c in collected_chunks if c.partial]
    assert len(partial_chunks) == 0
    
    # Should have tool_call chunks instead
    tool_chunks = [c for c in collected_chunks if c.tool_call]
    assert len(tool_chunks) >= 1


def test_handle_streaming_response_non_finish_tool():
    """Test streaming non-__finish__ tools don't create partial instances."""
    collected_chunks = []
    
    def callback(chunk):
        collected_chunks.append(chunk)
    
    def mock_stream():
        yield create_tool_delta_chunk(0, "call_1", "some_tool", "")
        yield create_tool_delta_chunk(0, None, None, '{"param": "value"}')
        yield create_finish_chunk()
    
    result = _handle_streaming_response(
        mock_stream(),
        callback,
        final_output_schema=TestOutput
    )
    
    # Should NOT have partial chunks (not __finish__)
    partial_chunks = [c for c in collected_chunks if c.partial]
    assert len(partial_chunks) == 0
    
    # Should have tool_call chunks
    tool_chunks = [c for c in collected_chunks if c.tool_call]
    assert len(tool_chunks) >= 1


def test_parse_partial_json_unit():
    """Unit test for _parse_partial_json function."""
    # Complete JSON
    result = _parse_partial_json('{"result": "test", "count": 5}', TestOutput)
    assert result.result == "test"
    assert result.count == 5
    
    # Incomplete JSON
    result = _parse_partial_json('{"result": "test"', TestOutput)
    assert result.result == "test"
    assert result.count is None
    
    # Empty string
    result = _parse_partial_json("", TestOutput)
    assert result is None


def test_streaming_with_progressive_json():
    """Test that partial JSON is parsed progressively during streaming."""
    collected_chunks = []
    
    def callback(chunk):
        collected_chunks.append(chunk)
    
    def mock_stream():
        yield create_tool_delta_chunk(0, "call_1", "__finish__", "")
        yield create_tool_delta_chunk(0, None, None, '{"result"')
        yield create_tool_delta_chunk(0, None, None, ': "test"')
        yield create_tool_delta_chunk(0, None, None, ', "count": 5}')
        yield create_finish_chunk()
    
    result = _handle_streaming_response(
        mock_stream(),
        callback,
        final_output_schema=TestOutput
    )
    
    # Should have multiple partial updates
    partial_chunks = [c for c in collected_chunks if c.partial]
    assert len(partial_chunks) >= 1
    
    # Last partial should have both fields
    last_partial = partial_chunks[-1].partial
    assert last_partial.result == "test"
    assert last_partial.count == 5


def test_call_llm_with_cache_none():
    """Test call_llm with cache=None (no parameter added)."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            cache=None
        )

        # Verify cache_control_injection_points was NOT added
        call_args = mock_completion.call_args
        assert "cache_control_injection_points" not in call_args.kwargs


def test_call_llm_with_cache_false():
    """Test call_llm with cache=False (no parameter added)."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            cache=False
        )

        # Verify cache_control_injection_points was NOT added
        call_args = mock_completion.call_args
        assert "cache_control_injection_points" not in call_args.kwargs


def test_call_llm_with_cache_true():
    """Test call_llm with cache=True (adds default array)."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            cache=True
        )

        # Verify cache_control_injection_points was added with default values
        call_args = mock_completion.call_args
        assert call_args.kwargs["cache_control_injection_points"] == [
            {"location": "message", "role": "system"},
            {"location": "message", "index": 0}
        ]


def test_call_llm_with_cache_custom():
    """Test call_llm with custom cache array."""
    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        mock_completion.return_value = MagicMock(
            choices=[MagicMock(
                message=MagicMock(content="test", tool_calls=[]),
                finish_reason="stop"
            )]
        )

        custom_cache = [{"location": "message", "role": "system"}]
        call_llm(
            messages=[{"role": "user", "content": "test"}],
            model="gpt-4",
            cache=custom_cache
        )

        # Verify cache_control_injection_points was forwarded as-is
        call_args = mock_completion.call_args
        assert call_args.kwargs["cache_control_injection_points"] == custom_cache
