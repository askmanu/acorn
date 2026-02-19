"""Example: Dependency Bloat & Overlap Scanner

Demonstrates the branching / map-reduce pattern in Acorn.

The DependencyScanner orchestrator fans out one PackageAnalyzerBranch per
dependency (map phase), collects the PackageProfile results, groups packages
by overlapping purpose, and returns a PruningPlan (reduce phase).

Supports both requirements.txt (pip) and package.json (npm) formats.
"""

import re

import requests
from pydantic import BaseModel, Field

from acorn import Module, tool


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class PackageInput(BaseModel):

    package_name: str = Field(description="Package name as it appears in the dependency file")

    ecosystem: str = Field(description="'pip' for PyPI packages, 'npm' for Node packages")

class PackageProfile(BaseModel):
    name: str = Field(description="Package name")
    primary_purpose: str = Field(description="One-line description of what the package does")
    categories: list[str] = Field(description="1-3 functional tags, e.g. ['http', 'networking']")
    key_features: list[str] = Field(description="3-5 bullet points describing key features")


class RedundancyGroup(BaseModel):

    purpose: str = Field(description="Shared purpose, e.g. 'HTTP client'")

    packages: list[str] = Field(description="the packages in this group")

    keep: str = Field(description="Recommended package to keep")

    remove: list[str] = Field(description="Recommended packages to remove")

    reason: str = Field(description="Why the recommended package wins")


class DependencyFileInput(BaseModel):
    file_content: str = Field(
        description="Contents of a requirements.txt or package.json file"
    )

class PruningPlan(BaseModel):

    summary: str = Field(description="Narrative paragraph summarising findings and recommendations")

    redundancy_groups: list[RedundancyGroup] = Field(description="Groups of overlapping packages")

    packages_to_remove: list[str] = Field(description="Flat list of packages recommended for removal")




# ---------------------------------------------------------------------------
# Tool (module-level so PackageAnalyzerBranch can reference it)
# ---------------------------------------------------------------------------


@tool
def fetch_package_info(package_name: str, ecosystem: str) -> dict:
    """Fetch README/description for a package from PyPI or the npm registry.

    Args:
        package_name: Package name as it appears in the dependency file
        ecosystem: "pip" for PyPI packages, "npm" for Node packages

    Returns:
        Dict with 'name', 'description', 'readme' (trimmed to 3 000 chars)
    """
    # Strip version pins: requests==2.31.0 → requests, axios@1.6.0 → axios
    clean_name = re.split(r"[=<>!@~^]", package_name)[0].strip()

    if ecosystem == "pip":
        url = f"https://pypi.org/pypi/{clean_name}/json"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            info = data.get("info", {})
            description = info.get("summary", "") or ""
            readme = info.get("description", "") or ""
        except requests.HTTPError:
            return {"name": clean_name, "description": "Not found on PyPI.", "readme": ""}
    else:  # npm
        url = f"https://registry.npmjs.org/{clean_name}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            description = data.get("description", "") or ""
            readme = data.get("readme", "") or ""
        except requests.HTTPError:
            return {"name": clean_name, "description": "Not found on npm.", "readme": ""}

    return {
        "name": clean_name,
        "description": description,
        "readme": readme[:3000],
    }


# ---------------------------------------------------------------------------
# Branch module — one instance per dependency
# ---------------------------------------------------------------------------


BRANCH_PROMPT = """Analyze a single package and determine its primary purpose.

1. Call fetch_package_info with the package_name and ecosystem.
2. From the description and README, identify: primary_purpose (one clear
   sentence), categories (1-3 functional tags), and key_features
   (3-5 bullet points).
3. Finish immediately after fetching — do not loop.
"""


class PackageAnalyzerBranch(Module):
    """Analyze a single package and extract its purpose and features."""

    system_prompt = BRANCH_PROMPT
    model = "anthropic/claude-sonnet-4-6"
    temperature = 0.2
    max_steps = 5

    initial_input = PackageInput
    final_output = PackageProfile

    tools = [fetch_package_info]


# ---------------------------------------------------------------------------
# Orchestrator module
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a dependency analyst specializing in finding redundant libraries.

Workflow:
1. Parse the file_content to extract all top-level dependency names.
   Detect the ecosystem: "npm" if it parses as JSON (package.json),
   "pip" if it looks like requirements.txt.

2. For EACH dependency use the PackageAnalyzerBranch branch to analyze each. You can analyze multiple in parallel.

3. After all branches finish, review the collected PackageProfile results.
   Group packages by overlapping primary_purpose / categories.

4. Identify RedundancyGroups — packages that do the same job.
   Recommend keeping the most popular/maintained one and removing the rest.

5. Call __finish__ with the complete PruningPlan.
"""


class DependencyScanner(Module):
    """Scan a requirements.txt or package.json for redundant dependencies."""

    system_prompt = SYSTEM_PROMPT
    model = "anthropic/claude-haiku-4-5"
    temperature = 0.2
    max_steps = 60

    initial_input = DependencyFileInput
    final_output = PruningPlan

    branches = [PackageAnalyzerBranch]
