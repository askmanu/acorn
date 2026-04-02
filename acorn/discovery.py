"""Tool discovery system for Acorn.

Provides a registry that indexes tools by name and description,
enabling search-based tool discovery instead of loading all tools
into the LLM prompt.
"""

import json
import re
from typing import Any

from acorn.tool_schema import generate_tool_schema


def _tokenize(text: str) -> set[str]:
    """Tokenize text into lowercase words for matching.

    Splits on non-alphanumeric characters and underscores.
    Filters out very short tokens (< 2 chars).
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if len(w) >= 2}


class ToolEntry:
    """A registered tool with its searchable metadata."""

    __slots__ = ("name", "description", "schema", "tool", "tokens")

    def __init__(self, name: str, description: str, schema: dict, tool: Any):
        self.name = name
        self.description = description
        self.schema = schema
        self.tool = tool
        # Pre-tokenize for search
        self.tokens = _tokenize(f"{name} {description}")


class ToolRegistry:
    """Indexes tools for search-based discovery.

    Provides keyword search over tool names and descriptions.
    No external dependencies — uses simple token overlap scoring.

    Example:
        >>> registry = ToolRegistry(tools)
        >>> results = registry.search("send email")
        >>> # Returns list of tool schemas matching the query
    """

    def __init__(self, tools: list):
        """Build the registry from a list of tools.

        Args:
            tools: List of callables with _tool_schema attributes.
        """
        self._entries: list[ToolEntry] = []
        self._by_name: dict[str, ToolEntry] = {}

        for tool in tools:
            schema = self._get_schema(tool)
            func_schema = schema.get("function", schema)
            name = func_schema.get("name", getattr(tool, "__name__", "unknown"))
            description = func_schema.get("description", "")

            entry = ToolEntry(
                name=name,
                description=description,
                schema=schema,
                tool=tool,
            )
            self._entries.append(entry)
            self._by_name[name] = entry

    @staticmethod
    def _get_schema(tool: Any) -> dict:
        """Get the tool schema, generating it if needed."""
        if hasattr(tool, "_tool_schema"):
            return tool._tool_schema
        return generate_tool_schema(tool)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        """Search tools by keyword matching on name and description.

        Scores each tool by token overlap between query and the tool's
        name + description. Returns top matches sorted by score.

        Args:
            query: Search query string
            limit: Maximum number of results to return

        Returns:
            List of tool schemas matching the query, sorted by relevance.
        """
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        scored = []
        for entry in self._entries:
            # Skip internal tools
            if entry.name.startswith("__"):
                continue

            # Score by exact token overlap
            overlap = query_tokens & entry.tokens
            score = len(overlap)

            # Also check prefix/substring matching for partial words
            # e.g., "email" should match "emails"
            if not overlap:
                for qt in query_tokens:
                    for et in entry.tokens:
                        if qt in et or et in qt:
                            score += 0.5
            else:
                # Bonus for partial matches on non-overlapping tokens
                remaining_query = query_tokens - overlap
                remaining_entry = entry.tokens - overlap
                for qt in remaining_query:
                    for et in remaining_entry:
                        if qt in et or et in qt:
                            score += 0.5

            if score > 0:
                # Bonus for name matches
                name_tokens = _tokenize(entry.name)
                for qt in query_tokens:
                    for nt in name_tokens:
                        if qt == nt:
                            score += 2
                        elif qt in nt or nt in qt:
                            score += 1
                scored.append((score, entry))

        # Sort by score descending, then by name for stability
        scored.sort(key=lambda x: (-x[0], x[1].name))

        results = []
        for _, entry in scored[:limit]:
            results.append(entry.schema)

        return results

    def get(self, name: str) -> dict | None:
        """Look up a tool schema by exact name.

        Args:
            name: The tool name to look up

        Returns:
            Tool schema dict, or None if not found.
        """
        entry = self._by_name.get(name)
        return entry.schema if entry else None

    def add(self, tool: Any) -> None:
        """Add a tool to the registry.

        Args:
            tool: A callable with _tool_schema attribute.
        """
        schema = self._get_schema(tool)
        func_schema = schema.get("function", schema)
        name = func_schema.get("name", getattr(tool, "__name__", "unknown"))
        description = func_schema.get("description", "")

        entry = ToolEntry(
            name=name,
            description=description,
            schema=schema,
            tool=tool,
        )
        self._entries.append(entry)
        self._by_name[name] = entry

    def remove(self, name: str) -> None:
        """Remove a tool from the registry by name.

        Args:
            name: The tool name to remove.
        """
        if name in self._by_name:
            entry = self._by_name.pop(name)
            self._entries = [e for e in self._entries if e.name != name]

    def list_all(self) -> list[dict]:
        """List all registered tools (name and description only).

        Returns:
            List of dicts with 'name' and 'description' keys.
        """
        results = []
        for entry in self._entries:
            if not entry.name.startswith("__"):
                results.append({
                    "name": entry.name,
                    "description": entry.description,
                })
        return results

    def __len__(self) -> int:
        return len(self._entries)
