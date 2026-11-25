"""Acorn - LLM agent framework with structured I/O.

Acorn is a Python library for building LLM agents with structured inputs and outputs,
heavily influenced by DSPy.
"""

from pathlib import Path

from acorn._version import __version__
from acorn.decorators import tool
from acorn.exceptions import (
    AcornError,
    BranchError,
    ParseError,
    ServiceError,
    ToolConflictError,
    TruncatedOutputError,
)
from acorn.module import Module
from acorn.partial import Partial
from acorn.serialization import pydantic_to_xml, xml_to_pydantic
from acorn.service import Service
from acorn.template import Template
from acorn.types import Step, StreamChunk, ToolCall, ToolResult

__all__ = [
    "__version__",
    "Module",
    "Template",
    "tool",
    "Partial",
    "Path",
    "Step",
    "ToolCall",
    "ToolResult",
    "StreamChunk",
    "Service",
    "AcornError",
    "ParseError",
    "BranchError",
    "ServiceError",
    "ToolConflictError",
    "TruncatedOutputError",
    "pydantic_to_xml",
    "xml_to_pydantic",
]
