"""Tests for Partial model generation."""

import pytest
from pydantic import BaseModel, ValidationError
from acorn.partial import Partial


class SimplePerson(BaseModel):
    """Simple test model."""
    name: str
    age: int


class NestedModel(BaseModel):
    """Model with nested fields."""
    title: str
    person: SimplePerson
    tags: list[str]


def test_partial_simple_model():
    """Test Partial with a simple model."""
    PartialPerson = Partial(SimplePerson)

    # All fields should be optional
    p1 = PartialPerson()
    assert p1.name is None
    assert p1.age is None

    p2 = PartialPerson(name="Alice")
    assert p2.name == "Alice"
    assert p2.age is None

    p3 = PartialPerson(age=30)
    assert p3.name is None
    assert p3.age == 30

    p4 = PartialPerson(name="Bob", age=25)
    assert p4.name == "Bob"
    assert p4.age == 25


def test_partial_nested_model():
    """Test Partial with nested models."""
    PartialNested = Partial(NestedModel)

    # All fields optional
    n1 = PartialNested()
    assert n1.title is None
    assert n1.person is None
    assert n1.tags is None

    n2 = PartialNested(title="Test")
    assert n2.title == "Test"
    assert n2.person is None


def test_partial_preserves_validation():
    """Test that Partial still validates field types."""
    PartialPerson = Partial(SimplePerson)

    # Valid types should work
    p = PartialPerson(name="Alice", age=30)
    assert p.name == "Alice"
    assert p.age == 30

    # Type coercion should work
    p2 = PartialPerson(age="25")
    assert p2.age == 25


def test_partial_non_model_raises():
    """Test that Partial raises TypeError for non-models."""
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="Partial\\(\\) requires a Pydantic BaseModel"):
        Partial(NotAModel)


def test_partial_field_access():
    """Test accessing fields on Partial models."""
    PartialPerson = Partial(SimplePerson)

    p = PartialPerson(name="Charlie")

    # Accessing present field
    assert p.name == "Charlie"

    # Accessing absent field returns None
    assert p.age is None


def test_partial_model_dump():
    """Test dumping Partial model to dict."""
    PartialPerson = Partial(SimplePerson)

    p = PartialPerson(name="Diana")
    data = p.model_dump()

    assert data == {"name": "Diana", "age": None}


def test_partial_excludes_none():
    """Test excluding None values from dump."""
    PartialPerson = Partial(SimplePerson)

    p = PartialPerson(name="Eve")
    data = p.model_dump(exclude_none=True)

    assert data == {"name": "Eve"}


def test_partial_name():
    """Test that Partial generates appropriate model name."""
    PartialPerson = Partial(SimplePerson)
    assert "Partial" in PartialPerson.__name__
    assert "SimplePerson" in PartialPerson.__name__
