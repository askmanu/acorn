"""Convert Pydantic models to XML strings."""

import xml.etree.ElementTree as ET
from typing import Any, get_origin, get_args
from pydantic import BaseModel
from datetime import datetime, date
from enum import Enum


def pydantic_to_xml(
    model: BaseModel,
    root_tag: str = "input",
    include_descriptions: bool = True,
    description_format: str = "attribute",  # "attribute" or "comment"
) -> str:
    """Convert a Pydantic model to an XML string.

    Args:
        model: The Pydantic model instance to convert
        root_tag: The tag name for the root element
        include_descriptions: Whether to include field descriptions
        description_format: How to include descriptions ("attribute" or "comment")

    Returns:
        Pretty-printed XML string

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> person = Person(name="Alice", age=30)
        >>> xml = pydantic_to_xml(person)
        >>> print(xml)
        <input>
          <name>Alice</name>
          <age>30</age>
        </input>
    """
    if not isinstance(model, BaseModel):
        raise TypeError(f"Expected Pydantic BaseModel instance, got {type(model)}")

    root = ET.Element(root_tag)
    _model_to_element(root, model, include_descriptions, description_format)

    # Pretty print
    _indent(root)
    return ET.tostring(root, encoding="unicode", method="xml")


def _model_to_element(
    parent: ET.Element,
    model: BaseModel,
    include_descriptions: bool,
    description_format: str,
) -> None:
    """Convert a Pydantic model to XML elements under parent."""
    for field_name, field_info in type(model).model_fields.items():
        value = getattr(model, field_name)

        # Skip None values for optional fields (but not empty lists)
        if value is None:
            continue

        # Include empty lists/dicts as empty elements
        if isinstance(value, (list, dict)) and len(value) == 0:
            # Create empty element
            ET.SubElement(parent, field_name)
            continue

        # Create element for this field
        field_element = ET.SubElement(parent, field_name)

        # Add description if requested
        if include_descriptions and field_info.description:
            if description_format == "attribute":
                field_element.set("description", field_info.description)
            elif description_format == "comment":
                comment = ET.Comment(f" {field_info.description} ")
                parent.insert(list(parent).index(field_element), comment)

        # Convert value to XML
        _value_to_element(field_element, value, include_descriptions, description_format)


def _value_to_element(
    element: ET.Element,
    value: Any,
    include_descriptions: bool,
    description_format: str,
) -> None:
    """Convert a value to XML content within element."""
    if value is None:
        # Empty element for None
        return

    elif isinstance(value, BaseModel):
        # Nested Pydantic model
        _model_to_element(element, value, include_descriptions, description_format)

    elif isinstance(value, dict):
        # Dictionary - each key becomes a sub-element
        for key, val in value.items():
            item_element = ET.SubElement(element, str(key))
            _value_to_element(item_element, val, include_descriptions, description_format)

    elif isinstance(value, (list, tuple)):
        # List/tuple - each item becomes an <item> sub-element
        for item in value:
            item_element = ET.SubElement(element, "item")
            _value_to_element(item_element, item, include_descriptions, description_format)

    elif isinstance(value, datetime):
        # ISO format for datetime
        element.text = value.isoformat()

    elif isinstance(value, date):
        # ISO format for date
        element.text = value.isoformat()

    elif isinstance(value, Enum):
        # Enum value
        element.text = str(value.value)

    elif isinstance(value, bool):
        # Boolean as lowercase string
        element.text = str(value).lower()

    elif isinstance(value, (int, float)):
        # Numeric primitives
        element.text = str(value)

    elif isinstance(value, str):
        # String - preserve XML content if present
        if '<' in value:
            try:
                parsed = ET.fromstring(f"<_wrapper>{value}</_wrapper>")
                element.text = parsed.text
                for child in parsed:
                    element.append(child)
                return
            except ET.ParseError:
                pass
        element.text = value

    else:
        # Fallback to string representation
        element.text = str(value)


def _indent(element: ET.Element, level: int = 0) -> None:
    """Add pretty-printing indentation to XML tree.

    Modifies the tree in-place by adding text and tail attributes.
    """
    indent_str = "\n" + "  " * level
    if len(element):  # Has children
        if not element.text or not element.text.strip():
            element.text = indent_str + "  "
        if not element.tail or not element.tail.strip():
            element.tail = indent_str
        for child in element:
            _indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = indent_str
    else:  # Leaf element
        if level and (not element.tail or not element.tail.strip()):
            element.tail = indent_str
