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

**Simple Q&A** ([simple_qa.py](simple_qa.py))  
The simplest Acorn example: a single-turn question-answering agent. Takes a question and returns a structured answer with confidence level. No tools, no agentic loop—just structured input and output.

### Agentic

**HN Production Readiness** ([hn_production_check.py](hn_production_check.py))  
Determines if a trending Hacker News project is production-ready or just a cool demo. Checks GitHub maturity markers (LICENSE, CONTRIBUTING, tests, CI/CD) and mines HN comments for technical concerns. Returns a Production Readiness Grade (A–F).

**Documentation Coverage** ([doc_coverage.py](doc_coverage.py))  
Scores the documentation coverage of a GitHub repository (0–100). Uses the GitHub recursive file tree to identify project type and complexity, checks for standard docs files (README, LICENSE, CONTRIBUTING, docs/), then reads key source files to measure inline docstring coverage. Returns a score, the top 5 most important undocumented functions, and a narrative summary.

### Branching

**Bus Factor Calculator** ([bus_factor.py](bus_factor.py))  
Calculates the "bus factor" of a GitHub repository: how many key contributors would need to stop before the project is in serious trouble. Demonstrates basic branching concepts.

**Dependency Bloat Scanner** ([dependency_scanner.py](dependency_scanner.py))  
Scans a requirements.txt or package.json for redundant and overlapping libraries. Spawns one `PackageAnalyzerBranch` per dependency (map phase) to fetch package metadata from PyPI or npm, then synthesizes a `PruningPlan` that groups packages by purpose and recommends which ones to remove (reduce phase). Demonstrates the branching / map-reduce pattern in Acorn.

## Adding New Examples

To add a new demo module to the web interface:

1. Create your module in `examples/your_module.py`
2. Register it in `examples/demo/registry.py` in the `DEMO_MODULES` dict:

```python
"your_module": {
    "title": "Your Module Title",
    "module_class": YourModuleClass,
    "description": "What your module does...",
    "category": "Basic",  # or "Agentic" or "Branching"
    "source_file": "your_module.py",
    "default_inputs": {
        "field_name": "default value"
    },
}
```

3. Run the app locally to test

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
