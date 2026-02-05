"""LiteLLM client wrapper for Acorn."""

import litellm
import json
from typing import Any, Iterator, Callable
from pydantic import BaseModel
from acorn.types import StreamChunk


def call_llm(
    messages: list[dict],
    model: str | dict,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tool_choice: str | dict | None = None,
    stream: bool = False,
    on_stream: Callable[[StreamChunk], None] | None = None,
    final_output_schema: type[BaseModel] | None = None,
    metadata: dict | None = None,
    cache: bool | list[dict] | None = None,
) -> dict:
    """Wrapper around litellm.completion for consistent LLM calls.

    Args:
        messages: List of message dictionaries (role, content)
        model: Model identifier string or config dict with keys:
               - id: model name (required)
               - vertex_location: Vertex AI location (optional)
               - vertex_credentials: Vertex AI credentials (optional)
               - reasoning: True or "low"/"medium"/"high" (optional)
        tools: Optional list of tool schemas
        temperature: Sampling temperature
        max_tokens: Maximum tokens to generate
        tool_choice: Optional tool choice directive
        stream: Whether to stream the response
        on_stream: Optional callback for streaming chunks
        final_output_schema: Optional Pydantic model for structured output
                            (used for partial streaming of __finish__ calls)
        metadata: Optional metadata dict for LiteLLM tracking
        cache: Optional caching configuration:
               - None: no caching (default)
               - False: no caching (same as None)
               - True: use default cache strategy (system message + first user message)
               - list[dict]: custom cache control injection points

    Returns:
        LiteLLM response dictionary (accumulated from stream if streaming)

    Raises:
        Exception: If the LLM call fails
    """
    # Build kwargs for litellm
    if isinstance(model, str):
        kwargs = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    else:
        # Model is a dict
        kwargs = {
            "model": model["id"],
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        # Add optional Vertex AI parameters
        if "vertex_location" in model:
            kwargs["vertex_location"] = model["vertex_location"]
        if "vertex_credentials" in model:
            kwargs["vertex_credentials"] = model["vertex_credentials"]

        # Handle reasoning parameter
        if "reasoning" in model:
            reasoning = model["reasoning"]
            if reasoning is True:
                kwargs["reasoning_effort"] = "medium"
            else:
                kwargs["reasoning_effort"] = reasoning

    # Add tools if provided
    if tools:
        kwargs["tools"] = tools

    # Add tool_choice if provided
    if tool_choice:
        kwargs["tool_choice"] = tool_choice

    # Add stream if requested
    if stream:
        kwargs["stream"] = True

    # Add metadata if provided
    if metadata:
        kwargs["metadata"] = metadata

    # Add cache control if provided
    if cache is not None:
        if cache is True:
            # Use default caching strategy
            kwargs["cache_control_injection_points"] = [
                {"location": "message", "role": "system"},
                {"location": "message", "index": 0}
            ]
        elif cache is not False:
            # Use custom cache configuration
            kwargs["cache_control_injection_points"] = cache

    # Call LiteLLM
    try:
        if stream and on_stream:
            # Streaming mode with callback
            response = litellm.completion(**kwargs)
            return _handle_streaming_response(response, on_stream, final_output_schema)
        elif stream:
            # Streaming mode without callback (just accumulate)
            response = litellm.completion(**kwargs)
            return _accumulate_streaming_response(response)
        else:
            # Non-streaming mode
            response = litellm.completion(**kwargs)
            return _response_to_dict(response)

    except Exception as e:
        # Re-raise with context
        raise Exception(f"LLM call failed: {e}") from e


def _response_to_dict(response: Any) -> dict:
    """Convert LiteLLM response to standardized dict format.

    Args:
        response: LiteLLM response object

    Returns:
        Dictionary with standardized response format
    """
    # Get the message from the response
    message = response.choices[0].message

    result = {
        "role": "assistant",
        "content": getattr(message, "content", None),
    }

    # Add tool calls if present
    if hasattr(message, "tool_calls") and message.tool_calls:
        result["tool_calls"] = []
        for tc in message.tool_calls:
            result["tool_calls"].append({
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }
            })

    # Add finish reason
    result["finish_reason"] = response.choices[0].finish_reason

    return result


def _handle_streaming_response(
    response: Iterator,
    on_stream: Callable[[StreamChunk], None],
    final_output_schema: type[BaseModel] | None = None,
) -> dict:
    """Handle streaming response with callback.

    Args:
        response: LiteLLM streaming response iterator
        on_stream: Callback to call for each chunk
        final_output_schema: Optional Pydantic model for partial streaming

    Returns:
        Accumulated response dictionary
    """
    accumulated_content = ""
    accumulated_tool_calls = []
    finish_reason = None

    for chunk in response:
        # Get delta from chunk
        delta = chunk.choices[0].delta if chunk.choices else None

        if not delta:
            continue

        # Handle content streaming
        if hasattr(delta, "content") and delta.content:
            accumulated_content += delta.content
            # Call callback with content chunk
            stream_chunk = StreamChunk(content=delta.content, done=False)
            on_stream(stream_chunk)

        # Handle tool call streaming
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                # Accumulate tool calls
                idx = tc_delta.index if hasattr(tc_delta, "index") else 0

                # Extend accumulated_tool_calls if needed
                while len(accumulated_tool_calls) <= idx:
                    accumulated_tool_calls.append({
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    })

                # Update accumulated tool call
                if hasattr(tc_delta, "id") and tc_delta.id:
                    accumulated_tool_calls[idx]["id"] = tc_delta.id

                if hasattr(tc_delta, "function"):
                    if hasattr(tc_delta.function, "name") and tc_delta.function.name:
                        accumulated_tool_calls[idx]["function"]["name"] = tc_delta.function.name
                    if hasattr(tc_delta.function, "arguments") and tc_delta.function.arguments:
                        accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

                # Check if this is a __finish__ call with schema
                tool_name = accumulated_tool_calls[idx]["function"]["name"]
                is_finish = tool_name == "__finish__"

                if is_finish and final_output_schema:
                    # Try to parse partial JSON for __finish__ calls
                    accumulated_args = accumulated_tool_calls[idx]["function"]["arguments"]
                    partial_instance = _parse_partial_json(accumulated_args, final_output_schema)

                    if partial_instance:
                        # Send partial structured output
                        stream_chunk = StreamChunk(partial=partial_instance, done=False)
                        on_stream(stream_chunk)
                    else:
                        # Parsing failed, send tool_call delta as fallback
                        stream_chunk = StreamChunk(
                            tool_call={"index": idx, "delta": tc_delta},
                            done=False
                        )
                        on_stream(stream_chunk)
                else:
                    # Not __finish__ or no schema - send tool call delta
                    stream_chunk = StreamChunk(
                        tool_call={"index": idx, "delta": tc_delta},
                        done=False
                    )
                    on_stream(stream_chunk)

        # Get finish reason
        if hasattr(chunk.choices[0], "finish_reason") and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    # Send final chunk
    final_chunk = StreamChunk(done=True)
    on_stream(final_chunk)

    # Build accumulated response
    result = {
        "role": "assistant",
        "content": accumulated_content if accumulated_content else None,
        "finish_reason": finish_reason
    }

    # Add tool calls if any
    if accumulated_tool_calls:
        result["tool_calls"] = accumulated_tool_calls

    return result


def _accumulate_streaming_response(response: Iterator) -> dict:
    """Accumulate streaming response without callback.

    Args:
        response: LiteLLM streaming response iterator

    Returns:
        Accumulated response dictionary
    """
    accumulated_content = ""
    accumulated_tool_calls = []
    finish_reason = None

    for chunk in response:
        # Get delta from chunk
        delta = chunk.choices[0].delta if chunk.choices else None

        if not delta:
            continue

        # Handle content
        if hasattr(delta, "content") and delta.content:
            accumulated_content += delta.content

        # Handle tool calls
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index if hasattr(tc_delta, "index") else 0

                # Extend accumulated_tool_calls if needed
                while len(accumulated_tool_calls) <= idx:
                    accumulated_tool_calls.append({
                        "id": None,
                        "type": "function",
                        "function": {
                            "name": "",
                            "arguments": ""
                        }
                    })

                # Update accumulated tool call
                if hasattr(tc_delta, "id") and tc_delta.id:
                    accumulated_tool_calls[idx]["id"] = tc_delta.id

                if hasattr(tc_delta, "function"):
                    if hasattr(tc_delta.function, "name") and tc_delta.function.name:
                        accumulated_tool_calls[idx]["function"]["name"] = tc_delta.function.name
                    if hasattr(tc_delta.function, "arguments") and tc_delta.function.arguments:
                        accumulated_tool_calls[idx]["function"]["arguments"] += tc_delta.function.arguments

        # Get finish reason
        if hasattr(chunk.choices[0], "finish_reason") and chunk.choices[0].finish_reason:
            finish_reason = chunk.choices[0].finish_reason

    # Build accumulated response
    result = {
        "role": "assistant",
        "content": accumulated_content if accumulated_content else None,
        "finish_reason": finish_reason
    }

    # Add tool calls if any
    if accumulated_tool_calls:
        result["tool_calls"] = accumulated_tool_calls

    return result

def _parse_partial_json(
    json_string: str,
    model_class: type[BaseModel]
) -> BaseModel | None:
    """Parse potentially incomplete JSON into Partial[T] instance.

    Attempts to parse accumulated JSON arguments. If parsing fails,
    tries to extract valid JSON prefix using lenient strategies.

    Args:
        json_string: Accumulated JSON string (potentially incomplete)
        model_class: Pydantic model class (e.g., final_output)

    Returns:
        Partial[T] instance with available fields, or None if unparsable
    """
    from acorn.partial import Partial

    if not json_string.strip():
        return None

    # Create Partial version of model
    PartialModel = Partial(model_class)

    # Strategy 1: Try direct parse (works if JSON is complete so far)
    try:
        data = json.loads(json_string)
        return PartialModel(**data)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Extract valid JSON prefix
    # Try to find last complete field by truncating at last comma/brace
    for end_pos in range(len(json_string), 0, -1):
        # Try truncating and closing with }
        candidate = json_string[:end_pos].rstrip() + "}"

        try:
            data = json.loads(candidate)
            return PartialModel(**data)
        except (json.JSONDecodeError, ValueError):
            continue

    # Strategy 3: Empty partial (all fields None)
    return PartialModel()
