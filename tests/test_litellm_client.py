"""Tests for LiteLLM client configuration."""

import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from acorn.llm.litellm_client import call_llm, _handle_streaming_response, _parse_partial_json, _extract_embedded_tool_calls, _response_to_dict, _translate_model_to_litellm
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


# Tests for _extract_embedded_tool_calls

def test_extract_embedded_single_tool_call():
    """Test extracting a single embedded tool call from content."""
    content = """I'll search for that.
<tool_call>
search_web
<arg_key>query</arg_key><arg_value>python async</arg_value>
</tool_call>"""
    result = _extract_embedded_tool_calls(content)
    assert len(result) == 1
    assert result[0]["type"] == "function"
    assert result[0]["function"]["name"] == "search_web"
    assert result[0]["id"].startswith("embedded_")
    import json
    args = json.loads(result[0]["function"]["arguments"])
    assert args["query"] == "python async"


def test_extract_embedded_multiple_tool_calls():
    """Test extracting multiple embedded tool calls."""
    content = """<tool_call>
search_web
<arg_key>query</arg_key><arg_value>topic A</arg_value>
</tool_call>
<tool_call>
calculate
<arg_key>expression</arg_key><arg_value>2+2</arg_value>
<arg_key>precision</arg_key><arg_value>2</arg_value>
</tool_call>"""
    result = _extract_embedded_tool_calls(content)
    assert len(result) == 2
    assert result[0]["function"]["name"] == "search_web"
    assert result[1]["function"]["name"] == "calculate"
    import json
    args = json.loads(result[1]["function"]["arguments"])
    assert args["expression"] == "2+2"
    assert args["precision"] == "2"


def test_extract_embedded_from_reasoning_content():
    """Test extracting tool calls from reasoning_content field."""
    reasoning = """Let me use the tool.
<tool_call>
__finish__
<arg_key>answer</arg_key><arg_value>42</arg_value>
</tool_call>"""
    result = _extract_embedded_tool_calls(None, reasoning)
    assert len(result) == 1
    assert result[0]["function"]["name"] == "__finish__"


def test_extract_embedded_no_matches():
    """Test that no tool_call tags returns empty list."""
    assert _extract_embedded_tool_calls("Just regular text") == []
    assert _extract_embedded_tool_calls(None, None) == []
    assert _extract_embedded_tool_calls("", "") == []


def test_extract_embedded_malformed_xml():
    """Test graceful handling of malformed tool_call blocks."""
    content = """<tool_call>
</tool_call>
<tool_call>
search_web
<arg_key>query</arg_key><arg_value>valid</arg_value>
</tool_call>"""
    result = _extract_embedded_tool_calls(content)
    # First block is malformed (no tool name), should be skipped
    assert len(result) == 1
    assert result[0]["function"]["name"] == "search_web"


def test_response_to_dict_surfaces_reasoning_content():
    """Test that reasoning_content is included in response dict."""
    mock_message = MagicMock()
    mock_message.content = "Hello"
    mock_message.reasoning_content = "I thought about this carefully"
    mock_message.tool_calls = []

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = mock_message
    mock_response.choices[0].finish_reason = "stop"

    result = _response_to_dict(mock_response)
    assert result["reasoning_content"] == "I thought about this carefully"
    assert result["content"] == "Hello"


def test_response_to_dict_fallback_embedded():
    """Test that embedded tool calls are extracted when no native calls present."""
    mock_message = MagicMock()
    mock_message.content = """<tool_call>
__finish__
<arg_key>result</arg_key><arg_value>done</arg_value>
</tool_call>"""
    mock_message.reasoning_content = None
    mock_message.tool_calls = []

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = mock_message
    mock_response.choices[0].finish_reason = "stop"

    result = _response_to_dict(mock_response)
    assert "tool_calls" in result
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["function"]["name"] == "__finish__"


def test_response_to_dict_native_takes_precedence():
    """Test that native tool calls take precedence over embedded ones."""
    mock_tc = MagicMock()
    mock_tc.id = "call_123"
    mock_tc.function.name = "search"
    mock_tc.function.arguments = '{"q": "test"}'

    mock_message = MagicMock()
    mock_message.content = """<tool_call>
__finish__
<arg_key>result</arg_key><arg_value>done</arg_value>
</tool_call>"""
    mock_message.reasoning_content = None
    mock_message.tool_calls = [mock_tc]

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = mock_message
    mock_response.choices[0].finish_reason = "stop"

    result = _response_to_dict(mock_response)
    assert len(result["tool_calls"]) == 1
    # Should be the native call, not the embedded one
    assert result["tool_calls"][0]["id"] == "call_123"
    assert result["tool_calls"][0]["function"]["name"] == "search"


# Tests for _translate_model_to_litellm

def test_translate_model_string_becomes_dict():
    """Test that string models are converted to dicts with nulled provider keys."""
    result = _translate_model_to_litellm("gpt-4")
    assert result["model"] == "gpt-4"
    assert result["api_base"] is None
    assert result["api_key"] is None
    assert result["vertex_location"] is None
    assert result["vertex_credentials"] is None
    assert result["reasoning_effort"] is None


def test_translate_model_dict_basic():
    """Test dict model translates id to model and nulls provider keys."""
    result = _translate_model_to_litellm({"id": "gpt-4"})
    assert result["model"] == "gpt-4"
    # Provider keys explicitly set to None to prevent leaking from primary model
    assert result["api_base"] is None
    assert result["api_key"] is None
    assert result["vertex_location"] is None
    assert result["vertex_credentials"] is None
    assert result["reasoning_effort"] is None


def test_translate_model_dict_with_all_keys():
    """Test dict model translates all supported keys."""
    result = _translate_model_to_litellm({
        "id": "vertex_ai/gemini-pro",
        "vertex_location": "us-central1",
        "vertex_credentials": "creds.json",
        "api_key": "sk-test",
        "api_base": "https://api.example.com",
    })
    assert result["model"] == "vertex_ai/gemini-pro"
    assert result["vertex_location"] == "us-central1"
    assert result["vertex_credentials"] == "creds.json"
    assert result["api_key"] == "sk-test"
    assert result["api_base"] == "https://api.example.com"
    assert result["reasoning_effort"] is None


def test_translate_model_dict_reasoning_true():
    """Test reasoning=True maps to reasoning_effort=medium."""
    result = _translate_model_to_litellm({"id": "gpt-4", "reasoning": True})
    assert result["model"] == "gpt-4"
    assert result["reasoning_effort"] == "medium"


def test_translate_model_dict_reasoning_string():
    """Test reasoning string maps to reasoning_effort."""
    result = _translate_model_to_litellm({"id": "gpt-4", "reasoning": "high"})
    assert result["model"] == "gpt-4"
    assert result["reasoning_effort"] == "high"


# Tests for model_fallbacks in call_llm

def test_call_llm_with_model_fallbacks():
    """Test that model_fallbacks are translated and passed as fallbacks kwarg."""
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
            model_fallbacks=[
                "anthropic/claude-3-5-sonnet-20241022",
                {"id": "vertex_ai/gemini-pro", "vertex_location": "us-central1"},
            ],
        )

        call_args = mock_completion.call_args
        assert "fallbacks" in call_args.kwargs
        fallbacks = call_args.kwargs["fallbacks"]
        # String fallback is converted to dict with nulled provider keys
        assert fallbacks[0]["model"] == "anthropic/claude-3-5-sonnet-20241022"
        assert fallbacks[0]["api_base"] is None
        assert fallbacks[0]["reasoning_effort"] is None
        # Dict fallback preserves its own keys, nulls the rest
        assert fallbacks[1]["model"] == "vertex_ai/gemini-pro"
        assert fallbacks[1]["vertex_location"] == "us-central1"
        assert fallbacks[1]["api_base"] is None
        assert fallbacks[1]["api_key"] is None
        assert fallbacks[1]["reasoning_effort"] is None


def test_call_llm_without_model_fallbacks():
    """Test that no fallbacks kwarg is added when model_fallbacks is None."""
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
        )

        call_args = mock_completion.call_args
        assert "fallbacks" not in call_args.kwargs


def test_call_llm_with_empty_model_fallbacks():
    """Test that empty list model_fallbacks doesn't add fallbacks kwarg."""
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
            model_fallbacks=[],
        )

        call_args = mock_completion.call_args
        assert "fallbacks" not in call_args.kwargs


# Tests for new completion parameters

def test_call_llm_max_tokens_none_by_default():
    """Test that max_tokens is not added when None."""
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
        )

        call_args = mock_completion.call_args
        assert "max_tokens" not in call_args.kwargs


def test_call_llm_max_tokens_passed_when_set():
    """Test that max_tokens is passed when explicitly set."""
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
            max_tokens=2048,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["max_tokens"] == 2048


def test_call_llm_max_completion_tokens():
    """Test that max_completion_tokens is passed through."""
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
            max_completion_tokens=1024,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["max_completion_tokens"] == 1024


def test_call_llm_top_p():
    """Test that top_p is passed through."""
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
            top_p=0.9,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["top_p"] == 0.9


def test_call_llm_stop():
    """Test that stop is passed through."""
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
            stop=["\n", "END"],
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["stop"] == ["\n", "END"]


def test_call_llm_presence_penalty():
    """Test that presence_penalty is passed through."""
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
            presence_penalty=0.6,
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["presence_penalty"] == 0.6


def test_call_llm_stream_options():
    """Test that stream_options is passed through."""
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
            stream_options={"include_usage": True},
        )

        call_args = mock_completion.call_args
        assert call_args.kwargs["stream_options"] == {"include_usage": True}


def test_call_llm_none_params_not_added():
    """Test that None-valued new params are not added to kwargs."""
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
        )

        call_args = mock_completion.call_args
        for key in ("max_tokens", "max_completion_tokens", "top_p", "stop", "presence_penalty", "stream_options"):
            assert key not in call_args.kwargs
