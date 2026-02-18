"""Demo registry and model presets."""

from examples.bus_factor import BusFactorCalculator
from examples.doc_coverage import DocCoverageAnalyzer
from examples.hn_production_check import HNProductionChecker
from examples.simple_qa import SimpleQA


DEMO_MODULES = {
    "simple_qa": {
        "title": "Simple Q&A",
        "module_class": SimpleQA,
        "description": (
            "The simplest Acorn example: a single-turn question-answering agent. "
            "Takes a question and returns a structured answer with confidence level. "
            "No tools, no agentic loop—just structured input and output."
        ),
        "category": "Basic",
        "source_file": "simple_qa.py",
        "default_inputs": {
            "text": "What is the capital of France?"
        },
    },
    "hn_production_check": {
        "title": "HN Production Readiness",
        "module_class": HNProductionChecker,
        "description": (
            "Determines if a trending Hacker News project is production-ready or just a cool demo. "
            "Checks GitHub maturity markers (LICENSE, CONTRIBUTING, tests, CI/CD) and mines HN comments "
            "for technical concerns. Returns a Production Readiness Grade (A–F)."
        ),
        "category": "Agentic",
        "source_file": "hn_production_check.py",
        "default_inputs": {
            "story_url": "https://news.ycombinator.com/item?id=47018675"
        },
        "env_inputs": {
            "GITHUB_TOKEN": {
                "label": "GitHub Token (optional)",
                "description": "Personal access token — higher rate limits (60 → 5,000 req/hr)",
                "placeholder": "ghp_...",
            }
        },
    },
    "doc_coverage": {
        "title": "Documentation Coverage",
        "module_class": DocCoverageAnalyzer,
        "description": (
            "Scores the documentation coverage of a GitHub repository (0–100). "
            "Uses the GitHub recursive file tree to identify project type and complexity, "
            "checks for standard docs files (README, LICENSE, CONTRIBUTING, docs/), "
            "then reads key source files to measure inline docstring coverage. "
            "Returns a score, the top 5 most important undocumented functions, and a narrative summary."
        ),
        "category": "Agentic",
        "source_file": "doc_coverage.py",
        "default_inputs": {
            "repo_url": "https://github.com/pallets/flask"
        },
        "env_inputs": {
            "GITHUB_TOKEN": {
                "label": "GitHub Token (optional)",
                "description": "Personal access token — higher rate limits (60 → 5,000 req/hr)",
                "placeholder": "ghp_...",
            }
        },
    },
    "bus_factor": {
        "title": "Bus Factor Calculator",
        "module_class": BusFactorCalculator,
        "description": (
            "Calculates the 'bus factor' of a GitHub repository: how many key contributors "
            "would need to stop before the project is in serious trouble. "
        ),
        "category": "Branching",
        "source_file": "bus_factor.py",
        "default_inputs": {
            "repo_url": "https://github.com/pallets/flask"
        },
        "env_inputs": {
            "GITHUB_TOKEN": {
                "label": "GitHub Token (optional)",
                "description": "Personal access token — higher rate limits (60 → 5,000 req/hr)",
                "placeholder": "ghp_...",
            }
        },
    },
}


MODEL_PRESETS = {
    "GLM5 (modal)": {
        "id": "openai/zai-org/GLM-5-FP8",
        "api_base": "https://api.us-west-2.modal.direct/v1",
    },
    "Claude Haiku 4.5": "anthropic/claude-haiku-4-5-20251001",
    "Claude Sonnet 4.5": "anthropic/claude-sonnet-4-5-20250929",
    "Claude Opus 4.6": "anthropic/claude-opus-4-6",
    "GPT-4": "openai/gpt-4",
    "GPT-4 Turbo": "openai/gpt-4-turbo-preview",
}
