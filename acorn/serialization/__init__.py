"""XML serialization for Pydantic models."""

from acorn.serialization.xml_encoder import pydantic_to_xml
from acorn.serialization.xml_decoder import xml_to_pydantic

__all__ = [
    "pydantic_to_xml",
    "xml_to_pydantic",
]
