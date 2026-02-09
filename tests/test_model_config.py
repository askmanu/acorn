"""Tests for model configuration handling."""

import pytest
from unittest.mock import patch
from pydantic import BaseModel

from acorn import Module


def test_model_string():
    """Test model as string."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = "gpt-4"
        final_output = Output

    mod = TestModule()
    assert mod.model == "gpt-4"


def test_model_dict_with_id():
    """Test model as dict with required id."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {"id": "anthropic/claude-3-5-sonnet-20241022"}
        final_output = Output

    mod = TestModule()
    assert mod.model["id"] == "anthropic/claude-3-5-sonnet-20241022"


def test_model_dict_missing_id():
    """Test that dict without id raises error."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {"vertex_location": "us-central1"}
        final_output = Output

    with pytest.raises(ValueError, match="Model dict must contain 'id' key"):
        TestModule()


def test_model_dict_invalid_keys():
    """Test that invalid keys raise error."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "gpt-4",
            "invalid_key": "value"
        }
        final_output = Output

    with pytest.raises(ValueError, match="Invalid model config keys"):
        TestModule()


def test_model_dict_with_vertex_params():
    """Test model dict with Vertex AI parameters."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "vertex_ai/gemini-pro",
            "vertex_location": "us-central1",
            "vertex_credentials": "path/to/creds.json"
        }
        final_output = Output

    mod = TestModule()
    assert mod.model["vertex_location"] == "us-central1"
    assert mod.model["vertex_credentials"] == "path/to/creds.json"


@patch("acorn.module.litellm.supports_reasoning", return_value=True)
def test_model_dict_with_reasoning_true(mock_supports):
    """Test reasoning parameter as True."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "gpt-4",
            "reasoning": True
        }
        final_output = Output

    mod = TestModule()
    assert mod.model["reasoning"] is True
    mock_supports.assert_called_once_with(model="gpt-4")


@patch("acorn.module.litellm.supports_reasoning", return_value=True)
def test_model_dict_with_reasoning_string(mock_supports):
    """Test reasoning parameter as string."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "gpt-4",
            "reasoning": "high"
        }
        final_output = Output

    mod = TestModule()
    assert mod.model["reasoning"] == "high"
    mock_supports.assert_called_once_with(model="gpt-4")


def test_model_dict_with_invalid_reasoning():
    """Test that invalid reasoning value raises error."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "gpt-4",
            "reasoning": "invalid"
        }
        final_output = Output

    with pytest.raises(ValueError, match="reasoning must be True or one of"):
        TestModule()


@patch("acorn.module.litellm.supports_reasoning", return_value=True)
def test_model_dict_all_parameters(mock_supports):
    """Test model dict with all allowed parameters."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "vertex_ai/gemini-pro",
            "vertex_location": "us-central1",
            "vertex_credentials": "creds.json",
            "reasoning": "medium"
        }
        final_output = Output

    mod = TestModule()
    assert mod.model["id"] == "vertex_ai/gemini-pro"
    assert mod.model["vertex_location"] == "us-central1"
    assert mod.model["vertex_credentials"] == "creds.json"
    assert mod.model["reasoning"] == "medium"


@patch("acorn.module.litellm.supports_reasoning", return_value=False)
def test_model_dict_reasoning_unsupported_model(mock_supports):
    """Test that reasoning with unsupported model raises error."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "gpt-4",
            "reasoning": True
        }
        final_output = Output

    with pytest.raises(ValueError, match="does not support reasoning"):
        TestModule()
    mock_supports.assert_called_once_with(model="gpt-4")


@patch("acorn.module.litellm.supports_reasoning")
def test_model_dict_no_reasoning_skips_check(mock_supports):
    """Test that supports_reasoning is not called when reasoning is not set."""
    class Output(BaseModel):
        result: str

    class TestModule(Module):
        model = {
            "id": "gpt-4"
        }
        final_output = Output

    TestModule()
    mock_supports.assert_not_called()
