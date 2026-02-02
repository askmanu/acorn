"""Tests for LiteLLM client configuration."""

import pytest
from unittest.mock import patch, MagicMock

from acorn.llm.litellm_client import call_llm


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
