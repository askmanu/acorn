"""Demo registry and model presets."""

from examples.bus_factor import BusFactorCalculator
from examples.dependency_scanner import DependencyScanner
from examples.doc_coverage import DocCoverageAnalyzer
from examples.hn_production_check import HNProductionChecker
from examples.license_checker import LicenseCompatibilityChecker, ProjectLicense
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
    "license_checker": {
        "title": "License Compatibility",
        "module_class": LicenseCompatibilityChecker,
        "description": (
            "Checks whether your project's dependencies have compatible licenses. "
            "Fetches license info from PyPI or npm for each dependency, then cross-references "
            "against known compatibility rules to flag conflicts (e.g. MIT project depending on "
            "a GPL library). Returns a compliance report with conflicts, warnings, and a summary."
        ),
        "category": "Agentic",
        "source_file": "license_checker.py",
        "default_inputs": {
            "project_license": ProjectLicense.MIT,
            "file_content": (
                "requests==2.31.0\n"
                "flask==3.0.3\n"
                "numpy==1.26.4\n"
                "pandas==2.2.1\n"
                "scikit-learn==1.4.2\n"
            ),
        },
    },
    "dependency_scanner": {
        "title": "Dependency Bloat Scanner",
        "module_class": DependencyScanner,
        "description": (
            "Scans a requirements.txt or package.json for redundant and overlapping libraries. "
            "Spawns one PackageAnalyzerBranch per dependency (map phase) to fetch package metadata "
            "from PyPI or npm, then synthesises a PruningPlan that groups packages by purpose and "
            "recommends which ones to remove (reduce phase). "
            "Demonstrates the branching / map-reduce pattern in Acorn."
        ),
        "category": "Branching",
        "source_file": "dependency_scanner.py",
        "default_inputs": {
            "file_content": (
                "requests==2.31.0\n"
                "httpx==0.27.0\n"
                "urllib3==2.2.1\n"
                "fastapi==0.111.0\n"
                "flask==3.0.3\n"
                "pytest==8.2.0\n"
                "pytest-cov==5.0.0\n"
                "coverage==7.5.1\n"
                "pydantic==2.7.1\n"
                "python-dotenv==1.0.1\n"
            )
        },
    },
}


MODEL_PRESETS = {
    "GLM5 (modal)": {
        "id": "openai/zai-org/GLM-5-FP8",
        "api_base": "https://api.us-west-2.modal.direct/v1",
    },
    "Claude Haiku 4.5": "anthropic/claude-haiku-4-5-20251001",
    "Claude Sonnet 4.6": "anthropic/claude-sonnet-4-6",
    "Claude Opus 4.6": "anthropic/claude-opus-4-6",
    "GPT-5.2": "openai/gpt-5.2-2025-12-11",
    "GPT-5.2 Codex": "openai/gpt-5.2-codex",
}
