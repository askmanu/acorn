"""Utilities for introspecting Pydantic schemas and mapping to Gradio components."""

import json
from enum import Enum
from typing import Any, get_origin, get_args

import gradio as gr
from pydantic_core import PydanticUndefined

from acorn import Module


def get_input_schema(module_class: type[Module]) -> dict[str, dict]:
    """Extract input schema from a Module's initial_input."""
    if not module_class.initial_input:
        return {}

    schema = {}
    for field_name, field_info in module_class.initial_input.model_fields.items():
        schema[field_name] = {
            "type": field_info.annotation,
            "description": field_info.description or "",
            "required": field_info.is_required(),
            "default": None if field_info.default is PydanticUndefined else field_info.default,
        }
    return schema


def get_output_schema(module_class: type[Module]) -> dict[str, dict]:
    """Extract output schema from a Module's final_output."""
    if not module_class.final_output:
        return {}

    schema = {}
    for field_name, field_info in module_class.final_output.model_fields.items():
        schema[field_name] = {
            "type": field_info.annotation,
            "description": field_info.description or "",
        }
    return schema


def _unwrap_optional(python_type):
    """Unwrap Optional[X] / X | None to get X."""
    origin = get_origin(python_type)
    if origin is not None:
        origin_str = str(origin)
        if "Union" in origin_str or "UnionType" in origin_str:
            args = get_args(python_type)
            non_none = [a for a in args if a is not type(None)]
            if non_none:
                return non_none[0]
    return python_type


def create_input_component(field_name: str, field_schema: dict) -> gr.Component:
    """Map a schema field to a Gradio input component."""
    python_type = _unwrap_optional(field_schema["type"])
    description = field_schema["description"]
    required = field_schema["required"]
    default = field_schema["default"]

    label = f"{field_name}{' *' if required else ''}"
    info = description or None

    if isinstance(python_type, type) and issubclass(python_type, Enum):
        choices = [e.value for e in python_type]
        return gr.Dropdown(label=label, info=info, choices=choices, value=default.value if isinstance(default, Enum) else (default or choices[0]))
    elif python_type is str:
        return gr.Textbox(label=label, info=info, value=default or "", placeholder=f"Enter {field_name}...")
    elif python_type is int:
        return gr.Number(label=label, info=info, value=default or 0, precision=0)
    elif python_type is float:
        return gr.Number(label=label, info=info, value=default or 0.0)
    elif python_type is bool:
        return gr.Checkbox(label=label, info=info, value=default or False)
    elif get_origin(python_type) is list or python_type is list:
        return gr.Code(label=label, info=info or "Enter JSON array", language="json", value=json.dumps(default or [], indent=2))
    elif get_origin(python_type) is dict or python_type is dict:
        return gr.Code(label=label, info=info or "Enter JSON object", language="json", value=json.dumps(default or {}, indent=2))
    else:
        return gr.Textbox(label=label, info=info, value=str(default) if default else "")


def parse_input_value(value: Any, field_type: type) -> Any:
    """Convert a UI value to the correct Python type."""
    field_type = _unwrap_optional(field_type)

    if value is None or value == "":
        return None

    if field_type is str:
        return str(value)
    elif field_type is int:
        return int(value)
    elif field_type is float:
        return float(value)
    elif field_type is bool:
        return bool(value)
    elif get_origin(field_type) is list or field_type is list:
        return json.loads(value) if isinstance(value, str) else value
    elif get_origin(field_type) is dict or field_type is dict:
        return json.loads(value) if isinstance(value, str) else value
    return value
