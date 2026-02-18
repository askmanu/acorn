"""Example: Hacker News Themes Analyzer

This example demonstrates an agent that fetches the top stories from Hacker News,
analyzes them, and identifies common themes across the frontpage.

Uses the official Hacker News API (https://github.com/HackerNews/API).
"""

import requests
from pydantic import BaseModel, Field
from acorn import Module, tool


# Input schema
class Input(BaseModel):
    num_stories: int = Field(
        default=10,
        description="Number of top stories to analyze (1-30)"
    )


# Output schema
class Story(BaseModel):
    title: str = Field(description="Story title")
    url: str = Field(description="Story URL (or HN discussion URL if no external URL)")
    description: str = Field(description="Brief description or summary")


class Output(BaseModel):
    themes: str = Field(description="Short description of common themes found")
    stories: list[Story] = Field(description="List of analyzed stories")


SYSTEM_PROMPT = """You are an AI assistant that analyzes trending topics on Hacker News.

Your task is to fetch the top stories from the Hacker News frontpage,
examine their titles and content, and identify common themes or patterns.
You should provide a concise summary of the prevailing topics and trends,
along with details about each story including title, URL, and description."""


class HackerNewsThemes(Module):
    """Analyzes trending topics on Hacker News."""

    system_prompt = SYSTEM_PROMPT
    initial_input = Input
    final_output = Output

    # Configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.3
    max_steps = 20

    @tool
    def get_top_story_ids(self, limit: int = 10) -> list[int]:
        """Get the IDs of top stories from Hacker News.

        Args:
            limit: Number of story IDs to return (max 30)

        Returns:
            List of story IDs
        """
        limit = min(limit, 30)  # Cap at 30
        response = requests.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json",
            timeout=10
        )
        response.raise_for_status()
        return response.json()[:limit]

    @tool
    def get_story_details(self, story_id: int) -> dict:
        """Get details for a specific Hacker News story.

        Args:
            story_id: The HN story ID

        Returns:
            Dictionary with story details (title, url, by, score, time, type, text)
        """
        response = requests.get(
            f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        # Add HN discussion URL if no external URL
        if not data.get("url"):
            data["url"] = f"https://news.ycombinator.com/item?id={story_id}"

        return data
