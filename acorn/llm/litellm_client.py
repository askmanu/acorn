"""LiteLLM client wrapper for Acorn."""

import litellm
from typing import Any


def call_llm(
    messages: list[dict],
    model: str | dict,
    tools: list[dict] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 4096,
    tool_choice: str | dict | None = None,
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

    Returns:
        LiteLLM response dictionary

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

    # Call LiteLLM
    try:
        response = litellm.completion(**kwargs)

        # Convert to dict format
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
