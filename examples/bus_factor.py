"""Example: Bus Factor Calculator

Analyzes contributor distribution for a GitHub repository and checks
whether the primary contributor is still actively maintaining it.
"""

import os

import requests
from pydantic import BaseModel, Field

from acorn import Module, tool


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class RepoInput(BaseModel):
    repo_url: str = Field(description="GitHub repository URL, e.g. https://github.com/owner/repo")


class VitalityReport(BaseModel):
    bus_factor: int = Field(description="Minimum contributors who must become unavailable to stall the project")
    risk_level: str = Field(description="One of: Low, Medium, High, Critical")
    details: str = Field(description="Narrative summary covering contributor distribution and primary contributor's recent activity")
    verdict: str = Field(description="One-sentence verdict, e.g. 'Bus Factor: 1 — If @alice stops, this project stops.'")
    recommendations: str = Field(description="Actionable suggestions to improve project health")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an open source health analyst specializing in project sustainability.
Your job is to calculate the bus factor of a GitHub repository and assess its long-term health.

Workflow:
1. Take note of the owner / repo from the repo URL
2. Call fetch_contributors to get the contributor list with commit counts
3. Calculate each contributor's percentage of total commits
4. Determine the bus factor (fewest top contributors whose combined commits exceed 50% of all commits)
5. Call fetch_user_activity for the primary contributor to check if they are still active
6. Assign a risk level:
   - Low: bus_factor >= 3 and primary is active
   - Medium: bus_factor == 2, or bus_factor >= 3 but primary is inactive
   - High: bus_factor == 1 and primary is active
   - Critical: bus_factor == 1 and primary is inactive
7. Write a verdict and provide actionable recommendations
"""


class BusFactorCalculator(Module):
    """Calculates the bus factor and health of a GitHub repository."""

    system_prompt = SYSTEM_PROMPT
    model = "anthropic/claude-sonnet-4-6"
    temperature = 0.3
    max_steps = 10

    initial_input = RepoInput
    final_output = VitalityReport

    def __init__(self, github_token: str | None = None):
        super().__init__()
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")

    def _github_headers(self) -> dict:
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    @tool
    def fetch_contributors(self, repo_path: str) -> list[dict]:
        """Fetch contributor stats from the GitHub API.

        Args:
            repo_path: Repository in 'owner/repo' format (e.g. 'pallets/flask')

        Returns:
            List of dicts with 'username' and 'commit_count', sorted by commit count descending
        """
        url = f"https://api.github.com/repos/{repo_path}/contributors"
        response = requests.get(url, params={"per_page": 50}, headers=self._github_headers(), timeout=15)
        response.raise_for_status()

        contributors = [
            {"username": c["login"], "commit_count": c["contributions"]}
            for c in response.json()
        ]
        contributors.sort(key=lambda x: x["commit_count"], reverse=True)
        return contributors

    @tool
    def fetch_user_activity(self, username: str) -> dict:
        """Fetch recent public activity for a GitHub user.

        Args:
            username: GitHub username

        Returns:
            Dict with event_count, last_event_date, event_types, active_repos
        """
        events_url = f"https://api.github.com/users/{username}/events/public"
        response = requests.get(events_url, params={"per_page": 30}, headers=self._github_headers(), timeout=15)
        response.raise_for_status()
        events = response.json()

        if not events:
            return {"event_count": 0, "last_event_date": "N/A", "event_types": [], "active_repos": []}

        last_event_date = events[0].get("created_at", "N/A")
        if last_event_date != "N/A":
            last_event_date = last_event_date[:10]

        return {
            "event_count": len(events),
            "last_event_date": last_event_date,
            "event_types": list({e.get("type", "Unknown") for e in events}),
            "active_repos": list({e.get("repo", {}).get("name", "") for e in events if e.get("repo")})[:10],
        }
