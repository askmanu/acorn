"""Demo registry and model presets."""

from examples.bus_factor import BusFactorCalculator
from examples.hn_themes import HackerNewsThemes
from examples.research_assistant import ResearchAssistant
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
    "hn_themes": {
        "title": "Hacker News Themes",
        "module_class": HackerNewsThemes,
        "description": (
            "Fetches top stories from Hacker News and analyzes common themes. "
            "Demonstrates API integration and multi-step data gathering with tools. "
            "Returns structured output with story details and thematic analysis."
        ),
        "category": "Agentic",
        "source_file": "hn_themes.py",
        "default_inputs": {
            "num_stories": 10
        },
    },
    "research_assistant": {
        "title": "Research Assistant",
        "module_class": ResearchAssistant,
        "description": (
            "Multi-step research agent that searches the web, finds academic papers, "
            "and analyzes data to produce a structured report with sources and confidence levels."
        ),
        "category": "Agentic",
        "source_file": "research_assistant.py",
        "default_inputs": {
            "topic": "Large Language Models",
            "depth": "shallow",
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
