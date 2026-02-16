"""Example: Research Assistant with Agentic Loop

This example demonstrates a research assistant that can search for information,
analyze it, and produce a structured report using the agentic loop.
"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field
from acorn import Module, tool


class Depth(str, Enum):
    shallow = "shallow"
    deep = "deep"

# Input schema
class Input(BaseModel):
    topic: str = Field(description="The research topic to investigate")
    depth: Depth = Field(description="Research depth")

# Output schema
class Output(BaseModel):
    summary: str = Field(description="Summary of findings")
    key_points: list[str] = Field(description="Key points discovered")
    sources: list[str] = Field(description="Sources consulted")
    confidence: str = Field(description="Confidence level: low, medium, or high")


SYSTEM_PROMPT = """You are an AI research assistant that gathers and analyzes information.

Your task is to search for relevant information using available tools,
synthesize your findings into clear insights, and provide well-sourced answers.
Use the search tools to gather data, then analyze and summarize your findings
with appropriate confidence levels."""


class ResearchAssistant(Module):
    """Research assistant that gathers and analyzes information."""

    system_prompt = SYSTEM_PROMPT
    initial_input = Input
    final_output = Output

    # Configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.3
    max_steps = 5

    # Tools available to the agent
    @tool
    def search_web(self, query: str) -> dict:
        """Search the web for information.

        Args:
            query: The search query

        Returns:
            Dictionary with search results
        """
        # In a real implementation, this would call a search API
        return {
            "query": query,
            "results": [
                f"Result 1 for '{query}'",
                f"Result 2 for '{query}'",
                f"Result 3 for '{query}'"
            ]
        }

    @tool
    def search_papers(self, keywords: str) -> list:
        """Search academic papers.

        Args:
            keywords: Keywords to search for

        Returns:
            List of paper titles
        """
        # In a real implementation, this would call an academic API
        return [
            f"Paper about {keywords} - Study 1",
            f"Paper about {keywords} - Study 2"
        ]

    @tool
    def analyze_data(self, data: str) -> str:
        """Analyze collected data.

        Args:
            data: The data to analyze

        Returns:
            Analysis results
        """
        # Simplified analysis
        return f"Analysis of: {data[:100]}..."
