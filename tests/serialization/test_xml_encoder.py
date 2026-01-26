"""Tests for XML encoding (Pydantic → XML)."""

import pytest
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

from acorn.serialization import pydantic_to_xml


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    age: int


class OptionalFieldsModel(BaseModel):
    """Model with optional fields."""
    required: str
    optional: str | None = None


class NestedModel(BaseModel):
    """Model with nested structure."""
    title: str
    person: SimpleModel


class ListModel(BaseModel):
    """Model with list field."""
    tags: list[str]
    numbers: list[int]


class DictModel(BaseModel):
    """Model with dict field."""
    metadata: dict[str, str]


class Color(Enum):
    """Test enum."""
    RED = "red"
    BLUE = "blue"


class SpecialTypesModel(BaseModel):
    """Model with special types."""
    active: bool
    color: Color
    created: datetime
    birthday: date


class DescribedModel(BaseModel):
    """Model with field descriptions."""
    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years")


def test_simple_model_to_xml():
    """Test encoding a simple model."""
    model = SimpleModel(name="Alice", age=30)
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<input>" in xml
    assert "<name>Alice</name>" in xml
    assert "<age>30</age>" in xml
    assert "</input>" in xml


def test_optional_fields_present():
    """Test optional field when present."""
    model = OptionalFieldsModel(required="test", optional="value")
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<required>test</required>" in xml
    assert "<optional>value</optional>" in xml


def test_optional_fields_absent():
    """Test optional field when absent."""
    model = OptionalFieldsModel(required="test")
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<required>test</required>" in xml
    assert "<optional>" not in xml  # Should be omitted


def test_nested_model():
    """Test encoding nested models."""
    person = SimpleModel(name="Bob", age=25)
    model = NestedModel(title="Profile", person=person)
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<title>Profile</title>" in xml
    assert "<person>" in xml
    assert "<name>Bob</name>" in xml
    assert "<age>25</age>" in xml
    assert "</person>" in xml


def test_list_field():
    """Test encoding list fields."""
    model = ListModel(tags=["python", "ai", "ml"], numbers=[1, 2, 3])
    xml = pydantic_to_xml(model, include_descriptions=False)

    # Lists should use <item> tags
    assert "<tags>" in xml
    assert xml.count("<item>") >= 6  # 3 tags + 3 numbers
    assert "<item>python</item>" in xml
    assert "<item>1</item>" in xml


def test_dict_field():
    """Test encoding dict fields."""
    model = DictModel(metadata={"key1": "value1", "key2": "value2"})
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<metadata>" in xml
    assert "<key1>value1</key1>" in xml
    assert "<key2>value2</key2>" in xml


def test_boolean_values():
    """Test boolean encoding."""
    model = SpecialTypesModel(
        active=True,
        color=Color.RED,
        created=datetime(2024, 1, 15, 10, 30),
        birthday=date(1990, 5, 20)
    )
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<active>true</active>" in xml


def test_enum_values():
    """Test enum encoding."""
    model = SpecialTypesModel(
        active=True,
        color=Color.BLUE,
        created=datetime(2024, 1, 15),
        birthday=date(1990, 5, 20)
    )
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<color>blue</color>" in xml


def test_datetime_values():
    """Test datetime encoding."""
    dt = datetime(2024, 1, 15, 10, 30, 45)
    model = SpecialTypesModel(
        active=True,
        color=Color.RED,
        created=dt,
        birthday=date(1990, 5, 20)
    )
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "2024-01-15T10:30:45" in xml


def test_date_values():
    """Test date encoding."""
    model = SpecialTypesModel(
        active=False,
        color=Color.RED,
        created=datetime(2024, 1, 15),
        birthday=date(1990, 5, 20)
    )
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "<birthday>1990-05-20</birthday>" in xml


def test_custom_root_tag():
    """Test custom root tag."""
    model = SimpleModel(name="Test", age=1)
    xml = pydantic_to_xml(model, root_tag="custom", include_descriptions=False)

    assert "<custom>" in xml
    assert "</custom>" in xml
    assert "<input>" not in xml


def test_descriptions_as_attributes():
    """Test field descriptions as attributes."""
    model = DescribedModel(name="Alice", age=30)
    xml = pydantic_to_xml(model, description_format="attribute")

    assert 'description="The person\'s name"' in xml
    assert 'description="The person\'s age in years"' in xml


def test_descriptions_as_comments():
    """Test field descriptions as comments."""
    model = DescribedModel(name="Alice", age=30)
    xml = pydantic_to_xml(model, description_format="comment")

    assert "The person's name" in xml
    assert "The person's age in years" in xml
    assert "<!--" in xml


def test_no_descriptions():
    """Test excluding descriptions."""
    model = DescribedModel(name="Alice", age=30)
    xml = pydantic_to_xml(model, include_descriptions=False)

    assert "The person's name" not in xml
    assert 'description=' not in xml


def test_pretty_printing():
    """Test that output is pretty-printed with indentation."""
    model = NestedModel(title="Test", person=SimpleModel(name="Bob", age=25))
    xml = pydantic_to_xml(model, include_descriptions=False)

    # Should have newlines and indentation
    assert "\n" in xml
    assert "  " in xml


def test_non_model_raises_error():
    """Test that non-Pydantic objects raise TypeError."""
    with pytest.raises(TypeError, match="Expected Pydantic BaseModel instance"):
        pydantic_to_xml({"not": "a model"})


def test_empty_list():
    """Test encoding empty list."""
    model = ListModel(tags=[], numbers=[])
    xml = pydantic_to_xml(model, include_descriptions=False)

    # Empty lists should still have the container tag (may have space before /)
    assert "<tags>" in xml or "<tags />" in xml or "<tags/>" in xml
    assert "<numbers>" in xml or "<numbers />" in xml or "<numbers/>" in xml
