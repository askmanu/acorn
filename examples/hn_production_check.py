"""Example: HN Production Readiness Checker

Analyzes a Hacker News story URL to determine whether the linked GitHub project
is production-ready or just a "cool demo." Checks maturity markers and mines
community comments for technical concerns.
"""

import base64
import os

import requests
from pydantic import BaseModel, Field

from acorn import Module, tool

from typing import Optional


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class HNStoryInput(BaseModel):
    story_url: str = Field(description="Show HN: story URL")

class ProductionReadinessReport(BaseModel):
    grade: Optional[str] = Field(description="Production readiness grade: A, B, C, D, or F")
    markers: Optional[str] = Field(description="Which maturity markers were found (LICENSE, CONTRIBUTING, tests, CI/CD) and which were missing")
    community_sentiment: Optional[str] = Field(description="Brief characterization of community sentiment (e.g. 'cautiously optimistic') including the biggest technical concern surfaced in comments (or 'None surfaced' if positive)")
    summary: str = Field(description="2-3 sentence narrative verdict covering the project maturity, and overall production readiness")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a software due-diligence analyst evaluating whether a Hacker News project is production-ready.

Workflow:
1. Parse the HN story ID from the URL (?id=...)

2. Call fetch_hn_story to get the story details and extract the GitHub URL
    - If the story does not have a github project attached to it, exit early and mention that in summary

3. Parse owner/repo from the GitHub URL (handle trailing slashes and .git suffixes)

4. Explore the repository freely using list_folder and read_file. Use your judgement — good signals to look for:
   - Licensing: LICENSE, LICENSE.md, COPYING, etc. in the root
   - Contribution culture: CONTRIBUTING.md, CODE_OF_CONDUCT.md, detailed README
   - Test discipline: test/, tests/, spec/, __tests__/ directories; look inside to gauge depth
   - CI/CD: .github/workflows/, .circleci/, .travis.yml, Makefile with test targets
   - Code quality: pyproject.toml / package.json / Cargo.toml for dependency pinning,
     CHANGELOG or release tags, security policy (SECURITY.md)
   - Red flags: TODO/FIXME density in source files, empty test files, no recent activity
   You are not limited to these — read any file that helps you form a more accurate verdict.

5. Call fetch_hn_comments to gather community discussion

6. Scan comments for technical keywords: performance, security, bug, crash, memory, scale,
   production, unstable, slow, CVE — surface the biggest concern (or "None surfaced")

7. Assign a Production Readiness Grade (A-F):
   - A: strong maturity signals across the board, no critical concerns
   - B: mostly solid, minor gaps or concerns
   - C: notable gaps in quality signals, or community concerns worth flagging
   - D: serious red flags in code quality, missing fundamentals, or critical community concerns
   - F: no maturity signals whatsoever, or critical security/stability issues

8. Return the structured report:
   - grade: letter grade
   - markers: what you found and what was missing across licensing, tests, CI, contribution culture
   - community_sentiment: overall tone plus the biggest technical concern (or "None surfaced")
   - summary: 2-3 sentences covering the project's maturity, and production readiness verdict
"""


class HNProductionChecker(Module):
    """Determines if a trending HN project is production-ready or just a cool demo."""

    system_prompt = SYSTEM_PROMPT
    model = "anthropic/claude-sonnet-4-6"
    temperature = 0.3
    max_steps = 25

    initial_input = HNStoryInput
    final_output = ProductionReadinessReport

    def __init__(self, github_token: str | None = None):
        super().__init__()
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")

    def _github_headers(self) -> dict:
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    @tool
    def fetch_hn_story(self, story_id: int) -> dict:
        """Fetch details for a Hacker News story using the Algolia HN API.

        Args:
            story_id: The HN story ID parsed from the ?id= query parameter

        Returns:
            Dict with title, url (the GitHub URL), author, points, num_comments
        """
        url = f"https://hn.algolia.com/api/v1/items/{story_id}"
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        data = response.json()

        return {
            "title": data.get("title", ""),
            "url": data.get("url", ""),
            "author": data.get("author", ""),
            "points": data.get("points", 0),
            "num_comments": data.get("num_comments", 0),
        }

    @tool
    def list_folder(self, repo_path: str, folder_path: str | None = None) -> list[dict]:
        """List the contents of a folder in a GitHub repository.

        Args:
            repo_path: Repository in 'owner/repo' format (e.g. 'pallets/flask')
            folder_path: Path within the repo to list (e.g. 'src/utils'). Pass None to list the root.

        Returns:
            List of dicts with 'name', 'type' ('file' or 'dir'), and 'size' (bytes, 0 for dirs)
        """
        path = folder_path.strip("/") if folder_path else ""
        url = f"https://api.github.com/repos/{repo_path}/contents/{path}"
        r = requests.get(url, headers=self._github_headers(), timeout=15)
        r.raise_for_status()
        return [
            {"name": entry["name"], "type": entry["type"], "size": entry.get("size", 0)}
            for entry in r.json()
        ]

    @tool
    def read_file(self, repo_path: str, file_path: str) -> str:
        """Read the contents of a file in a GitHub repository.

        Args:
            repo_path: Repository in 'owner/repo' format (e.g. 'pallets/flask')
            file_path: Path to the file within the repo (e.g. 'README.md' or 'src/main.py')

        Returns:
            Decoded text content of the file
        """
        url = f"https://api.github.com/repos/{repo_path}/contents/{file_path.strip('/')}"
        r = requests.get(url, headers=self._github_headers(), timeout=15)
        r.raise_for_status()
        data = r.json()
        return base64.b64decode(data["content"]).decode("utf-8", errors="replace")

    @tool
    def fetch_hn_comments(self, story_id: int, max_comments: int = 50) -> list[dict]:
        """Fetch top comments for a Hacker News story via the Algolia search API.

        Args:
            story_id: The HN story ID
            max_comments: Maximum number of comments to return (default 50)

        Returns:
            List of dicts with 'author' and 'text' keys, filtered to non-empty comments
        """
        url = "https://hn.algolia.com/api/v1/search"
        params = {
            "tags": f"comment,story_{story_id}",
            "hitsPerPage": max_comments,
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        hits = response.json().get("hits", [])

        return [
            {"author": h.get("author", ""), "text": h.get("comment_text", "")}
            for h in hits
            if h.get("comment_text")
        ]
