"""Convert XML strings to Pydantic models."""

import xml.etree.ElementTree as ET
from typing import Any, get_origin, get_args
from pydantic import BaseModel, ValidationError
from datetime import datetime, date
from enum import Enum


def xml_to_pydantic(xml_string: str, model_class: type[BaseModel]) -> BaseModel:
    """Parse XML and validate against a Pydantic model.

    Args:
        xml_string: XML string to parse
        model_class: Pydantic model class to validate against

    Returns:
        Instance of model_class with data from XML

    Raises:
        ValidationError: If XML data doesn't match model schema
        ET.ParseError: If XML is malformed

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> xml = '<input><name>Alice</name><age>30</age></input>'
        >>> person = xml_to_pydantic(xml, Person)
        >>> person.name
        'Alice'
    """
    if not issubclass(model_class, BaseModel):
        raise TypeError(f"Expected Pydantic BaseModel class, got {model_class}")

    # Parse XML
    root = ET.fromstring(xml_string)

    # Convert to dict
    data = _element_to_dict(root, model_class)

    # Validate with Pydantic
    return model_class(**data)


def _element_to_dict(element: ET.Element, model_class: type[BaseModel] | None = None) -> dict:
    """Convert an XML element to a dictionary.

    Args:
        element: The XML element to convert
        model_class: Optional Pydantic model to guide type conversion

    Returns:
        Dictionary representation of the XML element
    """
    result = {}

    # Get field type hints if model provided
    field_types = {}
    if model_class is not None:
        for field_name, field_info in model_class.model_fields.items():
            field_types[field_name] = field_info.annotation

    # Process child elements
    for child in element:
        field_name = child.tag
        field_type = field_types.get(field_name) if field_types else None

        # Convert child element to value
        value = _element_to_value(child, field_type)

        # Handle duplicate tags (convert to list)
        if field_name in result:
            # Already exists - convert to list or append
            if not isinstance(result[field_name], list):
                result[field_name] = [result[field_name]]
            result[field_name].append(value)
        else:
            result[field_name] = value

    return result


def _element_to_value(element: ET.Element, field_type: type | None = None) -> Any:
    """Convert an XML element to a Python value.

    Args:
        element: The XML element to convert
        field_type: Optional type hint to guide conversion

    Returns:
        Python value (str, int, float, bool, list, dict, or Pydantic model)
    """
    # Check if element has children
    has_children = len(element) > 0
    has_text = element.text is not None and element.text.strip()

    if not has_children and not has_text:
        # Empty element - check if it should be an empty list
        if field_type is not None:
            origin = get_origin(field_type)
            if origin is list:
                return []
        # Otherwise None
        return None

    if has_children:
        # Check if all children are <item> tags (indicates a list)
        if all(child.tag == "item" for child in element):
            # Parse as list
            item_type = _get_list_item_type(field_type)
            return [_element_to_value(child, item_type) for child in element]

        # Check if this is a nested model
        elif field_type and _is_pydantic_model(field_type):
            # Parse as nested Pydantic model
            nested_data = _element_to_dict(element, field_type)
            return field_type(**nested_data)

        else:
            # Parse as dictionary
            return _element_to_dict(element)

    # Leaf element with text
    text = element.text.strip()

    # Type conversion based on field_type
    if field_type is None:
        # No type hint - return as string
        return text

    # Unwrap Optional types
    origin = get_origin(field_type)
    if origin is type(None) or str(origin) == 'typing.UnionType':
        # Optional[X] or X | None
        args = get_args(field_type)
        if args:
            # Use first non-None type
            field_type = next((arg for arg in args if arg is not type(None)), str)

    # Convert based on type
    if field_type is bool or field_type == bool:
        return text.lower() in ('true', '1', 'yes')
    elif field_type is int or field_type == int:
        return int(text)
    elif field_type is float or field_type == float:
        return float(text)
    elif field_type is str or field_type == str:
        return text
    elif _is_pydantic_model(field_type):
        # Nested model with only text content (shouldn't happen often)
        return field_type(**{element.tag: text})
    elif hasattr(field_type, '__origin__'):
        # Generic type - try to handle
        origin = get_origin(field_type)
        if origin is list:
            # Single-item list
            item_type = get_args(field_type)[0] if get_args(field_type) else str
            return [_convert_text(text, item_type)]

    # Fallback to string
    return text


def _convert_text(text: str, target_type: type) -> Any:
    """Convert text to target type."""
    if target_type is bool or target_type == bool:
        return text.lower() in ('true', '1', 'yes')
    elif target_type is int or target_type == int:
        return int(text)
    elif target_type is float or target_type == float:
        return float(text)
    else:
        return text


def _is_pydantic_model(field_type: type) -> bool:
    """Check if a type is a Pydantic model."""
    try:
        return isinstance(field_type, type) and issubclass(field_type, BaseModel)
    except TypeError:
        return False


def _get_list_item_type(field_type: type | None) -> type | None:
    """Extract the item type from a List type hint."""
    if field_type is None:
        return None

    origin = get_origin(field_type)
    if origin is list:
        args = get_args(field_type)
        return args[0] if args else None

    return None
