"""Module execution logic."""

import html as html_mod
import threading
import traceback
from queue import Queue

from .registry import DEMO_MODULES
from .schema_utils import get_input_schema, parse_input_value


def run_module(demo_key: str, model_name: str, api_key: str, env_vars: dict | None = None, **input_kwargs):
    """Execute a Module, yielding step updates as they happen.

    Yields:
        ("step", markdown_str) for each completed step
        ("done", (output_dict, error)) when execution finishes
    """
    config = DEMO_MODULES[demo_key]
    module_class = config["module_class"]
    input_schema = get_input_schema(module_class)

    # Parse inputs
    parsed_inputs = {}
    for field_name, field_schema in input_schema.items():
        if field_name in input_kwargs:
            parsed = parse_input_value(input_kwargs[field_name], field_schema["type"])
            if parsed is not None:
                parsed_inputs[field_name] = parsed

    step_queue = Queue()

    def _format_step(step, label="Step", indent="") -> list[str]:
        """Format a step into two HTML chunks: reasoning+calls, then results.

        Returns a list of 1-2 HTML strings so the UI can display the
        model's reasoning and tool invocations immediately, followed by
        a separate entry for the tool results.
        """
        prefix = f"{label} {step.counter}"
        response = step.response or {}
        thought = response.get("reasoning_content") or response.get("content") or ""

        # --- Part 1: reasoning + tool call requests ---
        call_lines = [f"{indent}[{prefix}]"]
        if thought:
            call_lines.append(f"{indent}  {thought.strip()}")
        for tc in step.tool_calls:
            args_str = str(tc.arguments)
            call_lines.append(f"{indent}  → {tc.name}({args_str})")

        print("\n".join(call_lines), flush=True)
        call_html_lines = [f"<b>{html_mod.escape(prefix)}</b>"] + [
            html_mod.escape(l.strip()) for l in call_lines[1:]
        ]
        chunks = ["<br>".join(call_html_lines)]

        # --- Part 2: tool results (if any) ---
        if step.tool_results:
            result_lines = [f"{indent}[{prefix} · Results]"]
            for result in step.tool_results:
                if result.error:
                    result_lines.append(f"{indent}  ⚠ {result.name}: \n{result.error}")
                else:
                    out = str(result.output)
                    result_lines.append(f"{indent}  ✓ {result.name}: \n{out}")

            print("\n".join(result_lines), flush=True)
            result_html_lines = [f"<b>{html_mod.escape(f'{prefix} · Results')}</b>"] + [
                html_mod.escape(l.strip()) for l in result_lines[1:]
            ]
            chunks.append("<br>".join(result_html_lines))

        return chunks

    def _make_branch_on_step(bc):
        original = bc.__dict__.get("on_step")

        def on_step(self, step):
            for html in _format_step(step, label=f"↳ {bc.__name__} · Step", indent="    "):
                step_queue.put(
                    f'<div style="margin-left:14px;border-left:3px solid #ccc;padding-left:8px">{html}</div>'
                )
            if original:
                return original(self, step)
            return step

        return on_step

    # Create module with API key and custom on_step for streaming
    if isinstance(model_name, dict):
        _model_config = {**model_name, "api_key": api_key.strip()}
    else:
        _model_config = {"id": model_name, "api_key": api_key.strip()}

    # Patch branch classes: user-selected model + step logging
    _patched_branches = []
    for bc in getattr(module_class, "branches", None) or []:
        patched = type(bc.__name__, (bc,), {
            "model": _model_config,
            "on_step": _make_branch_on_step(bc),
        })
        _patched_branches.append(patched)

    class CustomModule(module_class):
        model = _model_config
        branches = _patched_branches

        def on_step(self, step):
            for chunk in _format_step(step):
                step_queue.put(chunk)
            if hasattr(super(), "on_step"):
                return super().on_step(step)
            return step

    init_kwargs = {k.lower(): v for k, v in (env_vars or {}).items() if v}

    result_holder = [None, None]  # (output_dict, error)

    def _execute():
        try:
            instance = CustomModule(**init_kwargs)
            result = instance(**parsed_inputs)
            if result is None:
                result_holder[0] = {"result": "Completed (no structured output)"}
            else:
                result_holder[0] = result.model_dump()
        except Exception as e:
            result_holder[1] = f"**Error:** {e}\n\n```\n{traceback.format_exc()}```"
        step_queue.put(None)  # sentinel

    thread = threading.Thread(target=_execute)
    thread.start()

    while True:
        item = step_queue.get()
        if item is None:
            break
        yield ("step", item)

    thread.join()
    yield ("done", (result_holder[0], result_holder[1]))
