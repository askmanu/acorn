"""Page builders for the Gradio app."""

import os
from pathlib import Path

import gradio as gr

# Serve examples/ directory as static so logo.png is accessible
_examples_dir = str(Path(__file__).parent.parent)
gr.set_static_paths(paths=[_examples_dir])

from .registry import DEMO_MODULES, MODEL_PRESETS
from .schema_utils import get_input_schema, get_output_schema, create_input_component
from .runner import run_module


def _build_progress_html(step_items: list[str]) -> str:
    """Build an HTML log of step progress."""
    if not step_items:
        return ""
    rows = []
    for item in step_items:
        rows.append(
            f'<div style="padding:6px 0;border-bottom:1px solid #eee;'
            f'font-size:13px;color:#555;line-height:1.4">{item}</div>'
        )
    return (
        '<div style="max-height:300px;overflow-y:auto;padding:8px 12px;'
        'background:#fafafa;border-radius:6px;border:1px solid #eee;margin-top:8px">'
        + "".join(rows)
        + "</div>"
    )


def _format_value(val) -> str:
    """Format a Python value as readable markdown."""
    if isinstance(val, list):
        if not val:
            return "*empty*"
        return "\n".join(f"- {item}" for item in val)
    if isinstance(val, dict):
        if not val:
            return "*empty*"
        return "\n".join(f"- **{k}:** {v}" for k, v in val.items())
    return str(val)


def build_home_page():
    """Build the landing / index page listing all demos."""

    logo_path = str(Path(__file__).parent.parent / "logo.png")
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("""
                # acorn Framework — Interactive Demos

                Explore acorn's capabilities through interactive examples.
                Each demo runs a real Acorn Module with your inputs.

                ## About
                LLM agent framework with structured I/O.
                Built on Pydantic for schemas and LiteLLM for multi-provider access.

                Check the project on github: [askmanu/acorn](https://github.com/askmanu/acorn)
            """)
        with gr.Column(scale=1):
            gr.Image(logo_path, show_label=False, container=False, height=80, elem_classes=["logo-right"])

    # gr.Markdown("")

    gr.Markdown("---")

    # Build all demo cards
    cards_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px; margin-top: 20px;">'

    for demo_key, config in DEMO_MODULES.items():
        cards_html += f"""
        <a href="/{demo_key}" style="
            text-decoration: none;
            color: inherit;
            flex: 1 1 calc(25% - 16px);
            min-width: 280px;
            max-width: 400px;
        ">
            <div style="
                padding: 24px;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                cursor: pointer;
                transition: all 0.2s ease;
                background: white;
                height: 100%;
                box-sizing: border-box;
            " onmouseover="this.style.borderColor='#666'; this.style.boxShadow='0 2px 8px rgba(0,0,0,0.1)'"
               onmouseout="this.style.borderColor='#e0e0e0'; this.style.boxShadow='none'">
                <h3 style="margin: 0 0 8px 0; font-size: 20px; font-weight: 600;">{config['title']}</h3>
                <div style="color: #666; font-size: 12px; margin-bottom: 12px; font-weight: 500;">{config['category']}</div>
                <p style="margin: 0 0 12px 0; line-height: 1.5; color: #333;">{config['description']}</p>
                <code style="font-size: 12px; color: #666; background: #f5f5f5; padding: 2px 6px; border-radius: 3px;">examples/{config['source_file']}</code>
            </div>
        </a>
        """

    cards_html += '</div>'
    gr.HTML(cards_html)


def build_demo_page(demo_key: str):
    """Build a demo page with two-column layout: config+input | output."""
    config = DEMO_MODULES[demo_key]
    module_class = config["module_class"]
    input_schema = get_input_schema(module_class)
    output_schema = get_output_schema(module_class)
    env_inputs = config.get("env_inputs", {})

    gr.Markdown("[← Back to Home](/)")

    # Radio toggle
    config_mode = gr.Radio(
        choices=["Our config (rate-limited)", "Custom config"],
        value="Our config (rate-limited)",
        label="Configuration Mode",
        show_label=False,
    )

    # Default info column (shown by default)
    with gr.Column(visible=True) as default_group:
        gr.Dropdown(
            choices=["GLM5 (modal)"],
            value="GLM5 (modal)",
            label="Model Provider",
            interactive=False,
        )
        gr.Markdown("""Uses our shared hosted models and keys, **rate-limited**.

            **Note**: You might get `Error code: 502` sometimes.

            Switch to Custom to use your own API key and model for testing.
            """
        )

    # Custom config column (hidden by default)
    with gr.Column(visible=False) as custom_group:
        with gr.Row():
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_PRESETS.keys()),
                value="GLM5 (modal)",
                label="Model Provider",
                scale=1,
            )
            api_key_input = gr.Textbox(
                label="API Key (required)",
                type="password",
                placeholder="Enter your API key",
                scale=2,
            )
        env_components: dict[str, gr.Component] = {}
        for env_key, env_meta in env_inputs.items():
            with gr.Row():
                env_components[env_key] = gr.Textbox(
                    label=env_meta["label"],
                    type="password",
                    placeholder=env_meta.get("placeholder", ""),
                    info=env_meta.get("description", ""),
                )

        gr.Markdown("Use your own API keys and github token")

    def _toggle_mode(mode):
        if mode == "Our config (rate-limited)":
            return gr.update(visible=True), gr.update(visible=False)
        return gr.update(visible=False), gr.update(visible=True)

    config_mode.change(
        fn=_toggle_mode,
        inputs=[config_mode],
        outputs=[default_group, custom_group],
    )

    gr.Markdown(f"# {config['title']}")
    source_file = config["source_file"]
    source_url = f"https://github.com/askmanu/acorn/blob/main/examples/{source_file}"
    gr.Markdown(f"Code: [{source_file}]({source_url})")
    gr.Markdown(config["description"])

    # Two-column layout for input and output
    with gr.Row(equal_height=False):
        # ── Left column: inputs ──
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### Input")
                input_components: dict[str, gr.Component] = {}
                if not input_schema:
                    gr.Markdown("*No input parameters.*")
                else:
                    for field_name, field_schema in input_schema.items():
                        comp = create_input_component(field_name, field_schema)
                        if field_name in config.get("default_inputs", {}):
                            comp.value = config["default_inputs"][field_name]
                        input_components[field_name] = comp

            run_btn = gr.Button("Run", variant="primary", size="lg")
            step_progress = gr.HTML(value="")

        # ── Right column: outputs ──
        with gr.Column(scale=1):
            output_components: dict[str, gr.Markdown] = {}
            if output_schema:
                for field_name in output_schema:
                    gr.Markdown(f"**{field_name}**")
                    output_components[field_name] = gr.Markdown(value="")
            else:
                gr.Markdown("**result**")
                output_components["result"] = gr.Markdown(value="")

            error_display = gr.Markdown(value="", visible=False)

    # ── Wiring ──
    field_names = list(input_components.keys())
    output_field_names = list(output_components.keys())

    def _run(*args):
        config_mode_val = args[0]
        model_name = args[1]
        api_key = args[2]
        env_values = args[3 : 3 + len(env_components)]
        input_values = args[3 + len(env_components):]

        # First yield: disable button, clear progress
        yield [gr.update() for _ in output_field_names] + [
            gr.update(),
            gr.update(interactive=False),
            gr.update(value=""),
        ]

        if config_mode_val == "Our config (rate-limited)":
            api_key = os.environ.get("DEMO_API_KEY", "")
            model_id = MODEL_PRESETS["GLM5 (modal)"]
            env_vars = {k: os.environ.get(k, "") for k in env_components}
        else:
            env_vars = dict(zip(env_components.keys(), env_values))
            if not api_key or not api_key.strip():
                results = [gr.update(value="") for _ in output_field_names]
                results.append(gr.update(
                    value="**Error:** API Key is required. Please enter your API key.",
                    visible=True,
                ))
                results.append(gr.update(interactive=True))
                results.append(gr.update(value=""))
                yield results
                return
            model_id = MODEL_PRESETS.get(model_name, model_name)

        inputs = {}
        for i, name in enumerate(field_names):
            if i < len(input_values):
                inputs[name] = input_values[i]

        step_items = []
        for event_type, data in run_module(demo_key, model_id, api_key, env_vars=env_vars, **inputs):
            if event_type == "step":
                step_items.append(data)
                progress_html = _build_progress_html(step_items)
                yield [gr.update() for _ in output_field_names] + [
                    gr.update(value="", visible=False),
                    gr.update(interactive=False),
                    gr.update(value=progress_html),
                ]
            elif event_type == "done":
                output_dict, error = data
                progress_html = _build_progress_html(step_items)
                results = []
                if error:
                    for _ in output_field_names:
                        results.append(gr.update(value=""))
                    results.append(gr.update(value=error, visible=True))
                else:
                    for fname in output_field_names:
                        val = output_dict.get(fname, "")
                        results.append(gr.update(value=_format_value(val)))
                    results.append(gr.update(value="", visible=False))
                results.append(gr.update(interactive=True))
                results.append(gr.update(value=progress_html))
                yield results

    all_inputs = [config_mode, model_dropdown, api_key_input] + list(env_components.values()) + list(input_components.values())
    all_outputs = list(output_components.values()) + [error_display, run_btn, step_progress]

    run_btn.click(fn=_run, inputs=all_inputs, outputs=all_outputs)
