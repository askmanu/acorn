"""Example: License Compatibility Checker

Checks whether a project's dependencies have compatible licenses.
Fetches license info from PyPI or npm, then cross-references against
known compatibility rules to flag conflicts.
"""

import re
from enum import Enum

import requests
from pydantic import BaseModel, Field

from acorn import Module, tool

from typing import Optional


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class ProjectLicense(str, Enum):
    MIT = "MIT"
    APACHE_2_0 = "Apache-2.0"
    GPL_2_0 = "GPL-2.0"
    GPL_3_0 = "GPL-3.0"
    LGPL_2_1 = "LGPL-2.1"
    LGPL_3_0 = "LGPL-3.0"
    AGPL_3_0 = "AGPL-3.0"
    MPL_2_0 = "MPL-2.0"
    BSD_2_CLAUSE = "BSD-2-Clause"
    BSD_3_CLAUSE = "BSD-3-Clause"
    ISC = "ISC"
    UNLICENSE = "Unlicense"

class LicenseCheckInput(BaseModel):
    project_license: ProjectLicense = Field(description="The project's own license")
    file_content: str = Field(description="Contents of a requirements.txt or package.json file")

class LicenseComplianceReport(BaseModel):
    conflicts: Optional[str] = Field(description="Narrative listing any license conflicts found, or None if all clear")
    warnings: Optional[str] = Field(description="Edge cases or licenses that need manual review")
    dependency_licenses: Optional[str] = Field(description="List of each dependency and its detected license")
    summary: str = Field(description="2-3 sentence overall compliance verdict")


# ---------------------------------------------------------------------------
# Module
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a software license compliance analyst. Your job is to check whether a project's dependencies have licenses compatible with the project's own license.

Workflow:
1. Parse the file_content to extract all dependency names.
   Detect the ecosystem: if it parses as JSON it's a package.json (npm), otherwise it's requirements.txt (pip).

2. For EACH dependency, call the appropriate tool:
   - Python/pip dependencies → fetch_pypi_license
   - Node/npm dependencies → fetch_npm_license

3. After collecting all license info, cross-reference each dependency's license against the project's license using the compatibility reference below.

4. Flag any conflicts (hard incompatibilities) and warnings (licenses requiring manual review).

5. Call __finish__ with the complete compliance report.

## License Compatibility Reference

Permissive licenses (MIT, BSD-2-Clause, BSD-3-Clause, ISC, Unlicense) are compatible with almost everything.

Compatibility matrix (rows = project license, columns = dependency license):

| Project \\ Dep | MIT | BSD | ISC | Apache-2.0 | MPL-2.0 | LGPL-2.1 | LGPL-3.0 | GPL-2.0 | GPL-3.0 | AGPL-3.0 |
|---|---|---|---|---|---|---|---|---|---|---|
| MIT            | OK  | OK  | OK  | OK         | OK*     | OK*      | OK*      | CONFLICT | CONFLICT | CONFLICT |
| BSD            | OK  | OK  | OK  | OK         | OK*     | OK*      | OK*      | CONFLICT | CONFLICT | CONFLICT |
| ISC            | OK  | OK  | OK  | OK         | OK*     | OK*      | OK*      | CONFLICT | CONFLICT | CONFLICT |
| Apache-2.0     | OK  | OK  | OK  | OK         | OK*     | OK*      | OK*      | CONFLICT | OK       | CONFLICT |
| MPL-2.0        | OK  | OK  | OK  | OK         | OK      | OK*      | OK*      | CONFLICT | OK       | CONFLICT |
| LGPL-2.1       | OK  | OK  | OK  | OK         | OK      | OK       | OK       | OK       | CONFLICT | CONFLICT |
| LGPL-3.0       | OK  | OK  | OK  | OK         | OK      | OK       | OK       | CONFLICT | OK       | CONFLICT |
| GPL-2.0        | OK  | OK  | OK  | CONFLICT   | OK      | OK       | CONFLICT | OK       | CONFLICT | CONFLICT |
| GPL-3.0        | OK  | OK  | OK  | OK         | OK      | OK       | OK       | CONFLICT | OK       | CONFLICT |
| AGPL-3.0       | OK  | OK  | OK  | OK         | OK      | OK       | OK       | CONFLICT | OK       | OK       |

OK* = Compatible but with conditions:
- MPL-2.0: Modifications to MPL-licensed files must stay under MPL (file-level copyleft)
- LGPL: Must allow re-linking; static linking may require LGPL compliance steps

Key rules:
- GPL-2.0 and GPL-3.0 are NOT compatible with each other (unless "GPL-2.0-or-later")
- AGPL-3.0 dependencies infect network-accessible projects (even without distribution)
- "Unlicense" and "CC0" are public domain equivalent — compatible with everything
- Dual-licensed packages: use the more permissive option
- If a license is unknown or non-standard, flag it as a WARNING for manual review

When reporting:
- CONFLICT: Hard incompatibility — using this dependency violates the project license
- WARNING: Needs manual review (unknown license, dual-license, or conditional compatibility)
- For each dependency, note its name and detected license in dependency_licenses
"""


class LicenseCompatibilityChecker(Module):
    """Check whether a project's dependencies have compatible licenses."""

    system_prompt = SYSTEM_PROMPT
    model = "anthropic/claude-sonnet-4-6"
    temperature = 0.2
    max_steps = 15

    initial_input = LicenseCheckInput
    final_output = LicenseComplianceReport

    @tool
    def fetch_pypi_license(self, package_name: str) -> dict:
        """Fetch license information for a Python package from PyPI.

        Use this tool for dependencies from a requirements.txt file.
        Call it once per dependency with the package name (version pins are stripped automatically).

        Args:
            package_name: Python package name (e.g. "requests", "flask==3.0.3")

        Returns:
            Dict with 'name', 'license', and 'homepage'
        """
        clean_name = re.split(r"[=<>!~^]", package_name)[0].strip()

        url = f"https://pypi.org/pypi/{clean_name}/json"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            info = data.get("info", {})
            # PEP 639: prefer license_expression (SPDX), fall back to license, then classifiers
            license_str = info.get("license_expression", "") or ""
            if not license_str:
                license_str = info.get("license", "") or ""
            classifiers = info.get("classifiers", [])
            if not license_str or len(license_str) > 100:
                for c in classifiers:
                    if c.startswith("License :: OSI Approved ::"):
                        license_str = c.split("::")[-1].strip()
                        break
            homepage = info.get("home_page", "") or info.get("project_url", "") or ""
        except requests.HTTPError:
            return {"name": clean_name, "license": "Unknown (not found on PyPI)", "homepage": ""}

        return {
            "name": clean_name,
            "license": license_str or "Unknown",
            "homepage": homepage,
        }

    @tool
    def fetch_npm_license(self, package_name: str) -> dict:
        """Fetch license information for a Node.js package from the npm registry.

        Use this tool for dependencies from a package.json file.
        Call it once per dependency with the package name (version pins are stripped automatically).

        Args:
            package_name: npm package name (e.g. "express", "axios@1.6.0")

        Returns:
            Dict with 'name', 'license', and 'homepage'
        """
        clean_name = re.split(r"[@=<>!~^]", package_name)[0].strip()

        url = f"https://registry.npmjs.org/{clean_name}"
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            license_str = data.get("license", "") or ""
            homepage = data.get("homepage", "") or ""
        except requests.HTTPError:
            return {"name": clean_name, "license": "Unknown (not found on npm)", "homepage": ""}

        return {
            "name": clean_name,
            "license": license_str or "Unknown",
            "homepage": homepage,
        }
