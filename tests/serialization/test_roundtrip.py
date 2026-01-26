"""Tests for roundtrip XML serialization (Pydantic → XML → Pydantic)."""

import pytest
from pydantic import BaseModel, Field
from datetime import datetime, date
from enum import Enum

from acorn.serialization import pydantic_to_xml, xml_to_pydantic


class SimpleModel(BaseModel):
    """Simple test model."""
    name: str
    age: int


class OptionalModel(BaseModel):
    """Model with optional fields."""
    required: str
    optional: str | None = None
    count: int | None = None


class NestedModel(BaseModel):
    """Model with nested structure."""
    title: str
    person: SimpleModel


class ListModel(BaseModel):
    """Model with lists."""
    tags: list[str]
    scores: list[int]


class MixedModel(BaseModel):
    """Model with various field types."""
    name: str
    age: int
    active: bool
    tags: list[str]


def test_simple_roundtrip():
    """Test simple model roundtrip."""
    original = SimpleModel(name="Alice", age=30)
    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, SimpleModel)

    assert restored.name == original.name
    assert restored.age == original.age
    assert restored == original


def test_optional_fields_present_roundtrip():
    """Test optional fields when present."""
    original = OptionalModel(required="test", optional="value", count=42)
    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, OptionalModel)

    assert restored == original
    assert restored.optional == "value"
    assert restored.count == 42


def test_optional_fields_absent_roundtrip():
    """Test optional fields when absent."""
    original = OptionalModel(required="test")
    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, OptionalModel)

    assert restored == original
    assert restored.optional is None
    assert restored.count is None


def test_nested_model_roundtrip():
    """Test nested model roundtrip."""
    original = NestedModel(
        title="Profile",
        person=SimpleModel(name="Bob", age=25)
    )
    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, NestedModel)

    assert restored == original
    assert restored.person.name == "Bob"
    assert restored.person.age == 25


def test_list_roundtrip():
    """Test list fields roundtrip."""
    original = ListModel(
        tags=["python", "ai", "ml"],
        scores=[95, 87, 92]
    )
    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, ListModel)

    assert restored == original
    assert restored.tags == ["python", "ai", "ml"]
    assert restored.scores == [95, 87, 92]


def test_empty_list_roundtrip():
    """Test empty lists roundtrip."""
    original = ListModel(tags=[], scores=[])
    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, ListModel)

    assert restored.tags == []
    assert restored.scores == []


def test_mixed_types_roundtrip():
    """Test model with mixed types."""
    original = MixedModel(
        name="Test",
        age=25,
        active=True,
        tags=["a", "b", "c"]
    )
    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, MixedModel)

    assert restored == original
    assert isinstance(restored.age, int)
    assert isinstance(restored.active, bool)
    assert isinstance(restored.tags, list)


def test_roundtrip_with_descriptions():
    """Test roundtrip preserves data even with descriptions."""
    class DescribedModel(BaseModel):
        name: str = Field(description="Person's name")
        age: int = Field(description="Person's age")

    original = DescribedModel(name="Alice", age=30)

    # Encode with descriptions
    xml = pydantic_to_xml(original, description_format="attribute")

    # Decode should ignore descriptions
    restored = xml_to_pydantic(xml, DescribedModel)

    assert restored == original


def test_roundtrip_custom_root_tag():
    """Test roundtrip with custom root tag."""
    original = SimpleModel(name="Test", age=42)
    xml = pydantic_to_xml(original, root_tag="custom", include_descriptions=False)
    restored = xml_to_pydantic(xml, SimpleModel)

    assert restored == original


def test_multiple_roundtrips():
    """Test that multiple roundtrips preserve data."""
    original = NestedModel(
        title="Test",
        person=SimpleModel(name="Charlie", age=35)
    )

    # First roundtrip
    xml1 = pydantic_to_xml(original, include_descriptions=False)
    restored1 = xml_to_pydantic(xml1, NestedModel)

    # Second roundtrip
    xml2 = pydantic_to_xml(restored1, include_descriptions=False)
    restored2 = xml_to_pydantic(xml2, NestedModel)

    # Should still match original
    assert restored2 == original


def test_complex_nested_roundtrip():
    """Test complex nested structure."""
    class Address(BaseModel):
        street: str
        city: str

    class Company(BaseModel):
        name: str
        address: Address

    class Employee(BaseModel):
        name: str
        age: int
        company: Company

    original = Employee(
        name="Alice",
        age=30,
        company=Company(
            name="TechCorp",
            address=Address(street="123 Main St", city="Boston")
        )
    )

    xml = pydantic_to_xml(original, include_descriptions=False)
    restored = xml_to_pydantic(xml, Employee)

    assert restored == original
    assert restored.company.address.city == "Boston"
