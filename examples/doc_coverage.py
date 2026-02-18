"""Example: Documentation Coverage Score

Analyzes a GitHub repository and produces a Documentation Coverage Score (0–100)
plus the top 5 most important undocumented functions/classes.
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

class RepoInput(BaseModel):
    repo_url: str = Field(description="GitHub repository URL, e.g. https://github.com/owner/repo")


class DocCoverageReport(BaseModel):
    score: Optional[int] = Field(description="Documentation coverage score from 0 to 100")
    summary: str = Field(description="2-3 sentence narrative covering project type, overall documentation state, and the most impactful improvement to make")
    advice: Optional[str] = Field(
        description=(
            "Top 5 most important improvements that can be made to the repo to increase the score"
        )
    )


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a documentation analyst evaluating the documentation coverage of a GitHub repository.

Workflow:
1. Parse owner/repo from the repo URL (handle trailing slashes and .git suffixes)
2. Call get_file_tree to get a full list of all files in the repo
3. Identify the project type (Python library, JS framework, CLI tool, etc.) from
   file extensions and config files (pyproject.toml, package.json, Cargo.toml, go.mod, etc.)
4. Check for standard documentation files by scanning the tree:
   - README (any casing, .md/.rst/.txt)
   - LICENSE / COPYING
   - CONTRIBUTING.md
   - CHANGELOG / HISTORY
   - docs/ folder or doc config (mkdocs.yml, docs/conf.py, .readthedocs.yaml)
5. Assess project complexity: count source files by language, identify core modules
   (look for src/, lib/, the package directory matching the project name)
6. Read key source files — prioritize:
   - The main entry point (e.g. __init__.py, index.ts, main.rs, lib.rs)
   - Public API modules (files that appear to export symbols)
   - The largest/most central source files
   Read as many as needed to get a representative sample (aim for 5-10 files).
7. In each file, identify public functions, classes, and methods. Check whether
   they have docstrings (Python), JSDoc comments (JS/TS), or doc comments (Rust/Go).
   Count documented vs undocumented public symbols.
8. Calculate a Documentation Coverage Score (0–100):
   - Standard docs (up to 30 points):
     README present: 15 pts | LICENSE: 5 pts | CONTRIBUTING: 5 pts | docs/ folder or CHANGELOG: 5 pts
   - Inline documentation rate (up to 70 points):
     (documented public symbols / total public symbols) * 70
   Round to nearest integer.
9. Identify the top 5 most important undocumented symbols — prioritize:
   - Public-facing API functions / classes used by external consumers
   - __init__ / constructor methods of core classes
   - Functions with many parameters or complex signatures
   - Entry points and top-level exports
10. Return the structured report.
"""


class DocCoverageAnalyzer(Module):
    """Scores the documentation coverage of a GitHub repository."""

    system_prompt = SYSTEM_PROMPT
    model = "anthropic/claude-sonnet-4-6"
    temperature = 0.3
    max_steps = 20

    initial_input = RepoInput
    final_output = DocCoverageReport

    def __init__(self, github_token: str | None = None):
        super().__init__()
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")

    def _github_headers(self) -> dict:
        headers = {"Accept": "application/vnd.github+json"}
        if self.github_token:
            headers["Authorization"] = f"Bearer {self.github_token}"
        return headers

    @tool
    def get_file_tree(self, repo_path: str) -> list[str]:
        """Get a flat list of all file paths in a GitHub repository.

        Args:
            repo_path: Repository in 'owner/repo' format (e.g. 'pallets/flask')

        Returns:
            Flat list of all file paths in the repository (blobs only), truncated to 2000 entries
        """
        url = f"https://api.github.com/repos/{repo_path}/git/trees/HEAD?recursive=1"
        r = requests.get(url, headers=self._github_headers(), timeout=15)
        r.raise_for_status()
        tree = r.json().get("tree", [])
        paths = [entry["path"] for entry in tree if entry.get("type") == "blob"]
        return paths[:2000]

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
