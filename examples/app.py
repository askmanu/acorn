"""Web interface for Acorn demos.

Run with hot reload:  gradio examples/app.py
Run normally:         python examples/app.py
"""

import gradio as gr

from demo.registry import DEMO_MODULES
from demo.pages import build_home_page, build_demo_page


def build_app() -> gr.Blocks:
    interface = gr.Blocks(title="Acorn Demos", theme=gr.themes.Soft())

    with interface:
        # --- Main Page (Root /) ---
        build_home_page()

    # --- Individual Demo Pages ---
    for demo_key, config in DEMO_MODULES.items():
        with interface.route(config["title"], f"/{demo_key}"):
            build_demo_page(demo_key)

    return interface


app = build_app()   # module-level for `gradio examples/app.py` hot-reload

if __name__ == "__main__":
    app.launch(show_error=True)
