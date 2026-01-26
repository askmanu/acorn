"""Partial model generation for streaming structured outputs."""

from typing import TypeVar, get_origin
from pydantic import BaseModel, create_model


T = TypeVar('T', bound=BaseModel)


def Partial(model: type[T]) -> type[T]:
    """Create a version of a Pydantic model where all fields are Optional.

    This is used for streaming structured outputs, where fields may not
    all be present as the output is being generated.

    Args:
        model: A Pydantic BaseModel class

    Returns:
        A new Pydantic model class with all fields made Optional

    Example:
        >>> from pydantic import BaseModel
        >>> class Person(BaseModel):
        ...     name: str
        ...     age: int
        >>> PartialPerson = Partial(Person)
        >>> p = PartialPerson(name="Alice")  # age is optional
        >>> p.name
        'Alice'
        >>> p.age is None
        True
    """
    if not issubclass(model, BaseModel):
        raise TypeError(f"Partial() requires a Pydantic BaseModel, got {model}")

    # Create field definitions with all fields made optional
    field_definitions = {}
    for field_name, field_info in model.model_fields.items():
        # Make the field Optional by wrapping in Optional[] or using None default
        field_type = field_info.annotation
        field_definitions[field_name] = (
            field_type | None,
            None  # Default value
        )

    # Create new model with optional fields
    partial_model = create_model(
        f"Partial{model.__name__}",
        __base__=BaseModel,
        **field_definitions
    )

    return partial_model
