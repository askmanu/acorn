"""Tests for cache configuration validation."""

import pytest
from pydantic import BaseModel
from acorn import Module


class TestInput(BaseModel):
    text: str


class TestOutput(BaseModel):
    result: str


def test_cache_none():
    """Test that cache=None is valid (default)."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = None

    # Should not raise
    module = TestModule()
    assert module.cache is None


def test_cache_true():
    """Test that cache=True is valid."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = True

    # Should not raise
    module = TestModule()
    assert module.cache is True


def test_cache_false():
    """Test that cache=False is valid."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = False

    # Should not raise
    module = TestModule()
    assert module.cache is False


def test_cache_custom_array():
    """Test that custom cache array is valid."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [{"location": "message", "role": "system"}]

    # Should not raise
    module = TestModule()
    assert module.cache == [{"location": "message", "role": "system"}]


def test_cache_invalid_type_string():
    """Test that cache with string type raises ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = "invalid"

    with pytest.raises(ValueError, match="cache must be None, bool, or list\\[dict\\]"):
        TestModule()


def test_cache_invalid_type_int():
    """Test that cache with int type raises ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = 123

    with pytest.raises(ValueError, match="cache must be None, bool, or list\\[dict\\]"):
        TestModule()


def test_cache_empty_list():
    """Test that empty cache list raises ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = []

    with pytest.raises(ValueError, match="cache list cannot be empty"):
        TestModule()


def test_cache_non_dict_items():
    """Test that non-dict items in cache list raise ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = ["invalid"]

    with pytest.raises(ValueError, match="cache\\[0\\] must be a dict"):
        TestModule()


def test_cache_missing_location():
    """Test that missing 'location' key raises ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [{"role": "system"}]

    with pytest.raises(ValueError, match="cache\\[0\\] must have 'location' key"):
        TestModule()


def test_cache_invalid_location():
    """Test that invalid 'location' value raises ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [{"location": "invalid"}]

    with pytest.raises(ValueError, match="cache\\[0\\]\\['location'\\] must be 'message'"):
        TestModule()


def test_cache_invalid_keys():
    """Test that invalid keys raise ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [{"location": "message", "invalid_key": "value"}]

    with pytest.raises(ValueError, match="cache\\[0\\] has invalid keys.*Allowed"):
        TestModule()


def test_cache_invalid_role_type():
    """Test that non-string role raises ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [{"location": "message", "role": 123}]

    with pytest.raises(ValueError, match="cache\\[0\\]\\['role'\\] must be a string"):
        TestModule()


def test_cache_invalid_index_type():
    """Test that non-int index raises ValueError."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [{"location": "message", "index": "0"}]

    with pytest.raises(ValueError, match="cache\\[0\\]\\['index'\\] must be an int"):
        TestModule()


def test_cache_all_valid_keys():
    """Test that all valid keys can be used together."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [
            {"location": "message", "role": "system", "index": 0}
        ]

    # Should not raise
    module = TestModule()
    assert module.cache == [{"location": "message", "role": "system", "index": 0}]


def test_cache_multiple_entries():
    """Test that multiple cache entries are valid."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [
            {"location": "message", "role": "system"},
            {"location": "message", "index": 0},
            {"location": "message", "role": "user"}
        ]

    # Should not raise
    module = TestModule()
    assert len(module.cache) == 3


def test_cache_negative_index():
    """Test that negative indices are valid (supported by LiteLLM)."""
    class TestModule(Module):
        initial_input = TestInput
        final_output = TestOutput
        cache = [{"location": "message", "index": -1}]

    # Should not raise
    module = TestModule()
    assert module.cache == [{"location": "message", "index": -1}]
