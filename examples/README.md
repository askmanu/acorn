---
title: acorn
emoji: 🌰
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: "6.5.1"
app_file: app.py
pinned: false
---

# acorn Demos

Interactive demos for [acorn](https://github.com/askmanu/acorn) — an LLM agent framework with structured I/O built on Pydantic and LiteLLM.

## Live Demos

Try the examples online at [huggingface.co/spaces/askmanu/acorn](https://huggingface.co/spaces/askmanu/acorn).

## Running Locally

Install dependencies:

```bash
pip install -r examples/requirements.txt
```

Set your API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Optional: Set a GitHub token for higher API rate limits (60 → 5,000 requests/hour):

```bash
export GITHUB_TOKEN="ghp_..."
```

Run the Gradio interface:

```bash
gradio examples/app.py
```

Or:

```bash
python examples/app.py
```

The web interface launches at `http://localhost:7860`.

## Example Modules

### Basic
- [Simple Q&A](simple_qa.py)

### Agentic
- [HN Production Readiness](hn_production_check.py)
- [Documentation Coverage](doc_coverage.py)

### Branching
- [Bus Factor Calculator](bus_factor.py)
- [Dependency Bloat Scanner](dependency_scanner.py)

## Project Structure

```
examples/
├── app.py                    # Gradio web interface entry point
├── demo/
│   ├── registry.py          # Demo module registry and model presets
│   ├── pages.py             # UI page builders (home page, demo pages)
│   ├── runner.py            # Module execution logic
│   ├── schema_utils.py      # Schema introspection utilities
│   └── theme.py             # Gradio theme configuration
├── simple_qa.py             # Basic Q&A example
├── hn_production_check.py   # Agentic example
├── doc_coverage.py          # Agentic example
├── bus_factor.py            # Branching example
├── dependency_scanner.py    # Branching map-reduce example
├── logo.png                 # Static asset
└── requirements.txt         # Dependencies
```