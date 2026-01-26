"""Tests for exception classes."""

import pytest
from acorn.exceptions import (
    AcornError,
    ParseError,
    BranchError,
    ToolConflictError,
)


def test_acorn_error():
    """Test base AcornError exception."""
    error = AcornError("test error")
    assert str(error) == "test error"
    assert isinstance(error, Exception)


def test_parse_error_without_raw_output():
    """Test ParseError without raw output."""
    error = ParseError("validation failed")
    assert str(error) == "validation failed"
    assert error.raw_output is None
    assert isinstance(error, AcornError)


def test_parse_error_with_raw_output():
    """Test ParseError with raw output."""
    raw = {"field": "value"}
    error = ParseError("validation failed", raw_output=raw)
    assert str(error) == "validation failed"
    assert error.raw_output == raw


def test_branch_error():
    """Test BranchError exception."""
    error = BranchError("branch failed")
    assert str(error) == "branch failed"
    assert isinstance(error, AcornError)


def test_tool_conflict_error():
    """Test ToolConflictError exception."""
    error = ToolConflictError("duplicate tool name")
    assert str(error) == "duplicate tool name"
    assert isinstance(error, AcornError)


def test_exception_hierarchy():
    """Test that all exceptions inherit from AcornError."""
    assert issubclass(ParseError, AcornError)
    assert issubclass(BranchError, AcornError)
    assert issubclass(ToolConflictError, AcornError)


def test_exception_catching():
    """Test that exceptions can be caught as AcornError."""
    try:
        raise ParseError("test")
    except AcornError:
        pass  # Should catch it

    try:
        raise BranchError("test")
    except AcornError:
        pass  # Should catch it

    try:
        raise ToolConflictError("test")
    except AcornError:
        pass  # Should catch it
