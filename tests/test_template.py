"""Tests for Jinja2 Template support."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import BaseModel

from acorn.template import Template


FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


class TestTemplateInline:
    """Tests for inline string templates."""

    def test_basic_render(self):
        t = Template(template="Hello {{ name }}", args={"name": "world"})
        assert t.render() == "Hello world"

    def test_no_args(self):
        t = Template(template="Hello there")
        assert t.render() == "Hello there"

    def test_multiple_args(self):
        t = Template(template="{{ a }} and {{ b }}", args={"a": "X", "b": "Y"})
        assert t.render() == "X and Y"

    def test_jinja_features(self):
        t = Template(
            template="{% for i in items %}{{ i }} {% endfor %}",
            args={"items": ["a", "b", "c"]},
        )
        assert t.render() == "a b c "


class TestTemplateFile:
    """Tests for file-based templates."""

    def test_render_from_file(self):
        t = Template(path="fixtures/test_prompt.md", args={"name": "Alice", "role": "helpful"})
        assert t.render() == "Hello Alice! You are a helpful assistant."

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            Template(path="fixtures/nonexistent.md")


class TestTemplateValidation:
    """Tests for input validation."""

    def test_both_path_and_template_raises(self):
        with pytest.raises(ValueError, match="not both"):
            Template(path="fixtures/test_prompt.md", template="Hello")

    def test_neither_path_nor_template_raises(self):
        with pytest.raises(ValueError, match="exactly one"):
            Template()


class TestTemplateLazyRendering:
    """Tests for lazy rendering / args mutation."""

    def test_args_mutation_before_render(self):
        t = Template(template="Hello {{ name }}", args={"name": "before"})
        t.args["name"] = "after"
        assert t.render() == "Hello after"

    def test_args_can_be_replaced(self):
        t = Template(template="{{ x }}", args={"x": 1})
        t.args = {"x": 99}
        assert t.render() == "99"


class TestPathRelativeResolution:
    """Tests for relative Path resolution in _build_system_message()."""

    def test_relative_path_resolves_against_class_file(self, tmp_path):
        """Relative Path should resolve against the defining class's source file."""
        from acorn.module import Module

        # Write a prompt file next to the fixtures dir
        prompt_file = Path(__file__).resolve().parent / "fixtures" / "test_prompt.md"

        class TestAgent(Module):
            system_prompt = Path("fixtures/test_prompt.md")

        agent = TestAgent.__new__(TestAgent)
        agent.system_prompt = TestAgent.system_prompt
        msg = agent._build_system_message()
        assert "Hello" in msg["content"]

    def test_absolute_path_unchanged(self):
        from acorn.module import Module

        prompt_file = Path(__file__).resolve().parent / "fixtures" / "test_prompt.md"

        class TestAgent(Module):
            system_prompt = prompt_file

        agent = TestAgent.__new__(TestAgent)
        agent.system_prompt = TestAgent.system_prompt
        msg = agent._build_system_message()
        assert "Hello" in msg["content"]


class TestTemplateModuleIntegration:
    """Tests for integration with Module._build_system_message()."""

    def test_build_system_message_with_template(self):
        from acorn.module import Module

        class TestAgent(Module):
            system_prompt = Template(template="You are {{ role }}", args={"role": "a helper"})

        agent = TestAgent.__new__(TestAgent)
        agent.system_prompt = TestAgent.system_prompt
        msg = agent._build_system_message()
        assert msg["content"] == "You are a helper"

    def test_build_system_message_with_file_template(self):
        from acorn.module import Module

        class TestAgent(Module):
            system_prompt = Template(
                path="fixtures/test_prompt.md",
                args={"name": "Bot", "role": "coding"},
            )

        agent = TestAgent.__new__(TestAgent)
        agent.system_prompt = TestAgent.system_prompt
        msg = agent._build_system_message()
        assert msg["content"] == "Hello Bot! You are a coding assistant."
