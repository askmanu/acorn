"""Custom Gradio theme for Acorn demos."""

import gradio as gr

# --- Design Variables ---
BACKGROUND_COLOR = "#ede4d6"
ACCENT_COLOR = "#513424"
FONT_FAMILY = "Playfair Display"

# --- Custom Color Palette derived from ACCENT_COLOR ---
acorn_palette = gr.themes.Color(
    c50="#f5f0ec",
    c100="#e8ddd4",
    c200="#d4c4b5",
    c300="#b99e8a",
    c400="#9a7560",
    c500="#7a5440",
    c600="#654434",
    c700="#513424",
    c800="#3e271b",
    c900="#2c1b13",
    c950="#1a100b",
    name="acorn",
)

neutral_palette = gr.themes.Color(
    c50="#f7f3ee",
    c100="#ede4d6",
    c200="#e0d4c3",
    c300="#c9b8a4",
    c400="#b09a82",
    c500="#8c7a66",
    c600="#6b5d4d",
    c700="#4f4538",
    c800="#362f26",
    c900="#231f19",
    c950="#15130e",
    name="acorn_neutral",
)


def create_theme() -> gr.themes.Base:
    """Build and return the custom Acorn theme."""
    theme = gr.themes.Base(
        primary_hue=acorn_palette,
        neutral_hue=neutral_palette,
        font=[gr.themes.GoogleFont(FONT_FAMILY), "serif"],
        text_size=gr.themes.sizes.text_lg,
    ).set(
        body_text_size="14px",
        body_background_fill=BACKGROUND_COLOR,
        block_background_fill="#f7f3ee",
        block_border_width="1px",
        block_border_color="#d4c4b5",
        block_label_background_fill="#e8ddd4",
        block_title_text_color=ACCENT_COLOR,
        button_primary_background_fill=ACCENT_COLOR,
        button_primary_background_fill_hover="#654434",
        button_primary_text_color="#ffffff",
        input_background_fill="#ffffff",
        input_border_color="#d4c4b5",
    )
    return theme


acorn_theme = create_theme()
