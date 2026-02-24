"""Jinja2 template support for system prompts."""

import inspect
from pathlib import Path

from jinja2 import BaseLoader, Environment, FileSystemLoader


class Template:
    """A Jinja2 template for system prompts.

    Supports file-based or inline string templates with lazy rendering.

    Args:
        path: Path to a template file (relative to the caller's file).
        template: Inline Jinja2 template string.
        args: Template variables for rendering. Defaults to {}.

    Exactly one of ``path`` or ``template`` must be provided.
    """

    def __init__(self, *, path: str | None = None, template: str | None = None, args: dict | None = None):
        if path is not None and template is not None:
            raise ValueError("Specify exactly one of 'path' or 'template', not both.")
        if path is None and template is None:
            raise ValueError("Specify exactly one of 'path' or 'template'.")

        self.args = args or {}

        if path is not None:
            # Resolve relative to caller's file
            caller_frame = inspect.stack()[1]
            caller_file = caller_frame.filename
            caller_dir = Path(caller_file).resolve().parent
            self._resolved_path = (caller_dir / path).resolve()
            if not self._resolved_path.is_file():
                raise FileNotFoundError(f"Template file not found: {self._resolved_path}")
            self._template_string = None
        else:
            self._resolved_path = None
            self._template_string = template

    def render(self) -> str:
        """Render the template with stored args."""
        if self._resolved_path is not None:
            loader = FileSystemLoader(str(self._resolved_path.parent))
            env = Environment(loader=loader)
            tmpl = env.get_template(self._resolved_path.name)
        else:
            env = Environment(loader=BaseLoader())
            tmpl = env.from_string(self._template_string)

        return tmpl.render(**self.args)
