"""Memory service for persistent storage and retrieval."""

import json
import sqlite3
from typing import Optional

from acorn.decorators import tool
from acorn.service import Service


class Memory(Service):
    """Long-term memory storage and retrieval.

    Save, search, and manage persistent memory entries using SQLite.
    Each entry has a key, content, and optional tags for categorization.
    """

    def __init__(self, path: str = "./memory.db"):
        """Initialize the Memory service.

        Args:
            path: Path to the SQLite database file. Use ":memory:" for
                  an in-memory database (useful for testing).
        """
        self.path = path
        self._conn: Optional[sqlite3.Connection] = None

    async def setup(self):
        """Create the database connection and ensure tables exist."""
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                key TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                tags TEXT DEFAULT '[]',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        self._conn.commit()

    async def teardown(self):
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    async def health(self) -> bool:
        """Check if the database connection is active."""
        if self._conn is None:
            return False
        try:
            self._conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False

    @tool
    def save(self, key: str, content: str, tags: list[str] = None) -> str:
        """Save or update a memory entry.

        Args:
            key: Unique identifier for the memory
            content: The content to store
            tags: Optional tags for categorization
        """
        tags_json = json.dumps(tags or [])
        self._conn.execute(
            """INSERT INTO memories (key, content, tags, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET
                   content = excluded.content,
                   tags = excluded.tags,
                   updated_at = CURRENT_TIMESTAMP""",
            (key, content, tags_json),
        )
        self._conn.commit()
        return f"Saved memory: {key}"

    @tool
    def search(self, query: str, limit: int = 5) -> str:
        """Search memories by keyword matching on key, content, and tags.

        Args:
            query: Search query (matches against key, content, and tags)
            limit: Maximum number of results to return
        """
        cursor = self._conn.execute(
            """SELECT key, content, tags FROM memories
               WHERE key LIKE ? OR content LIKE ? OR tags LIKE ?
               ORDER BY updated_at DESC
               LIMIT ?""",
            (f"%{query}%", f"%{query}%", f"%{query}%", limit),
        )
        rows = cursor.fetchall()

        if not rows:
            return "No memories found."

        results = []
        for row in rows:
            tags = json.loads(row["tags"])
            entry = {"key": row["key"], "content": row["content"]}
            if tags:
                entry["tags"] = tags
            results.append(entry)

        return json.dumps(results, indent=2)

    @tool
    def delete(self, key: str) -> str:
        """Delete a memory entry.

        Args:
            key: The key of the memory to delete
        """
        cursor = self._conn.execute(
            "DELETE FROM memories WHERE key = ?", (key,)
        )
        self._conn.commit()

        if cursor.rowcount > 0:
            return f"Deleted memory: {key}"
        return f"Memory not found: {key}"

    @tool
    def list_all(self, limit: int = 20) -> str:
        """List all stored memories.

        Args:
            limit: Maximum number of entries to return
        """
        cursor = self._conn.execute(
            "SELECT key, content, tags FROM memories ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )
        rows = cursor.fetchall()

        if not rows:
            return "No memories stored."

        results = []
        for row in rows:
            tags = json.loads(row["tags"])
            entry = {"key": row["key"], "content": row["content"]}
            if tags:
                entry["tags"] = tags
            results.append(entry)

        return json.dumps(results, indent=2)
