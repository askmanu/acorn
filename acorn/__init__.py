"""Acorn - LLM agent framework with structured I/O.

Acorn is a Python library for building LLM agents with structured inputs and outputs,
heavily influenced by DSPy.
"""

from pathlib import Path

from acorn._version import __version__
from acorn.decorators import tool
from acorn.partial import Partial
from acorn.types import Step, ToolCall, ToolResult, StreamChunk
from acorn.exceptions import AcornError, ParseError, BranchError, ToolConflictError
from acorn.serialization import pydantic_to_xml, xml_to_pydantic
from acorn.module import Module

__all__ = [
    "__version__",
    "Module",
    "tool",
    "Partial",
    "Path",
    "Step",
    "ToolCall",
    "ToolResult",
    "StreamChunk",
    "AcornError",
    "ParseError",
    "BranchError",
    "ToolConflictError",
    "pydantic_to_xml",
    "xml_to_pydantic",
]
