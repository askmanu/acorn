"""Tests for XML decoding (XML → Pydantic)."""

import pytest
from pydantic import BaseModel, ValidationError
from datetime import datetime, date

from acorn.serialization import xml_to_pydantic


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    age: int


class OptionalModel(BaseModel):
    """Model with optional fields."""
    required: str
    optional: str | None = None


class NestedModel(BaseModel):
    """Model with nested structure."""
    title: str
    person: SimpleModel


class ListModel(BaseModel):
    """Model with list fields."""
    tags: list[str]


class BoolModel(BaseModel):
    """Model with boolean."""
    active: bool


def test_simple_model_from_xml():
    """Test decoding simple XML."""
    xml = """
    <input>
        <name>Alice</name>
        <age>30</age>
    </input>
    """
    model = xml_to_pydantic(xml, SimpleModel)

    assert model.name == "Alice"
    assert model.age == 30


def test_optional_field_present():
    """Test optional field when present."""
    xml = """
    <input>
        <required>test</required>
        <optional>value</optional>
    </input>
    """
    model = xml_to_pydantic(xml, OptionalModel)

    assert model.required == "test"
    assert model.optional == "value"


def test_optional_field_absent():
    """Test optional field when absent."""
    xml = """
    <input>
        <required>test</required>
    </input>
    """
    model = xml_to_pydantic(xml, OptionalModel)

    assert model.required == "test"
    assert model.optional is None


def test_nested_model_from_xml():
    """Test decoding nested models."""
    xml = """
    <input>
        <title>Profile</title>
        <person>
            <name>Bob</name>
            <age>25</age>
        </person>
    </input>
    """
    model = xml_to_pydantic(xml, NestedModel)

    assert model.title == "Profile"
    assert model.person.name == "Bob"
    assert model.person.age == 25


def test_list_from_xml():
    """Test decoding list fields."""
    xml = """
    <input>
        <tags>
            <item>python</item>
            <item>ai</item>
            <item>ml</item>
        </tags>
    </input>
    """
    model = xml_to_pydantic(xml, ListModel)

    assert model.tags == ["python", "ai", "ml"]


def test_boolean_true():
    """Test decoding boolean true."""
    xml = "<input><active>true</active></input>"
    model = xml_to_pydantic(xml, BoolModel)
    assert model.active is True


def test_boolean_false():
    """Test decoding boolean false."""
    xml = "<input><active>false</active></input>"
    model = xml_to_pydantic(xml, BoolModel)
    assert model.active is False


def test_boolean_variants():
    """Test various boolean representations."""
    # Test "1"
    xml1 = "<input><active>1</active></input>"
    model1 = xml_to_pydantic(xml1, BoolModel)
    assert model1.active is True

    # Test "0"
    xml0 = "<input><active>0</active></input>"
    model0 = xml_to_pydantic(xml0, BoolModel)
    assert model0.active is False


def test_type_coercion():
    """Test that types are properly coerced."""
    xml = """
    <input>
        <name>Test</name>
        <age>25</age>
    </input>
    """
    model = xml_to_pydantic(xml, SimpleModel)

    assert isinstance(model.age, int)
    assert model.age == 25


def test_validation_error_on_invalid_data():
    """Test that validation errors are raised for invalid data."""
    xml = """
    <input>
        <name>Test</name>
        <age>not_a_number</age>
    </input>
    """

    with pytest.raises((ValidationError, ValueError)):
        xml_to_pydantic(xml, SimpleModel)


def test_validation_error_on_missing_required():
    """Test validation error when required field is missing."""
    xml = """
    <input>
        <name>Test</name>
    </input>
    """

    with pytest.raises(ValidationError):
        xml_to_pydantic(xml, SimpleModel)


def test_ignores_description_attributes():
    """Test that description attributes are ignored."""
    xml = """
    <input>
        <name description="The person's name">Alice</name>
        <age description="The person's age">30</age>
    </input>
    """
    model = xml_to_pydantic(xml, SimpleModel)

    assert model.name == "Alice"
    assert model.age == 30


def test_ignores_xml_comments():
    """Test that XML comments are ignored."""
    xml = """
    <input>
        <!-- This is a comment -->
        <name>Alice</name>
        <!-- Another comment -->
        <age>30</age>
    </input>
    """
    model = xml_to_pydantic(xml, SimpleModel)

    assert model.name == "Alice"
    assert model.age == 30


def test_malformed_xml_raises_error():
    """Test that malformed XML raises ParseError."""
    xml = "<input><name>Test</name>"  # Missing closing tag

    with pytest.raises(Exception):  # ET.ParseError
        xml_to_pydantic(xml, SimpleModel)


def test_non_model_class_raises_error():
    """Test that non-Pydantic class raises TypeError."""
    xml = "<input><name>Test</name></input>"

    with pytest.raises(TypeError, match="Expected Pydantic BaseModel class"):
        xml_to_pydantic(xml, dict)


def test_empty_xml_elements():
    """Test handling empty XML elements."""
    xml = """
    <input>
        <required>test</required>
        <optional></optional>
    </input>
    """
    model = xml_to_pydantic(xml, OptionalModel)

    assert model.required == "test"
    assert model.optional is None


def test_whitespace_handling():
    """Test that whitespace is properly handled."""
    xml = """
    <input>
        <name>  Alice  </name>
        <age>  30  </age>
    </input>
    """
    model = xml_to_pydantic(xml, SimpleModel)

    assert model.name == "Alice"
    assert model.age == 30
