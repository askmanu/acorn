"""Tests for the Memory service."""

import json
import pytest

from acorn.services.memory import Memory


@pytest.fixture
async def memory():
    """Create an in-memory Memory service."""
    mem = Memory(path=":memory:")
    await mem.setup()
    yield mem
    await mem.teardown()


class TestMemoryLifecycle:
    @pytest.mark.asyncio
    async def test_setup_creates_table(self, memory):
        cursor = memory._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
        )
        assert cursor.fetchone() is not None

    @pytest.mark.asyncio
    async def test_health_when_connected(self, memory):
        assert await memory.health() is True

    @pytest.mark.asyncio
    async def test_health_when_disconnected(self):
        mem = Memory(path=":memory:")
        assert await mem.health() is False

    @pytest.mark.asyncio
    async def test_teardown_closes_connection(self, memory):
        await memory.teardown()
        assert memory._conn is None
        assert await memory.health() is False


class TestMemorySave:
    @pytest.mark.asyncio
    async def test_save_basic(self, memory):
        result = memory.save(key="test", content="hello world")
        assert "Saved memory: test" in result

    @pytest.mark.asyncio
    async def test_save_with_tags(self, memory):
        result = memory.save(key="test", content="hello", tags=["greeting", "test"])
        assert "Saved" in result

        # Verify tags stored
        cursor = memory._conn.execute("SELECT tags FROM memories WHERE key='test'")
        row = cursor.fetchone()
        tags = json.loads(row["tags"])
        assert tags == ["greeting", "test"]

    @pytest.mark.asyncio
    async def test_save_upsert(self, memory):
        memory.save(key="test", content="v1")
        memory.save(key="test", content="v2")

        cursor = memory._conn.execute("SELECT content FROM memories WHERE key='test'")
        assert cursor.fetchone()["content"] == "v2"


class TestMemorySearch:
    @pytest.mark.asyncio
    async def test_search_by_content(self, memory):
        memory.save(key="greeting", content="hello world")
        memory.save(key="farewell", content="goodbye world")

        result = memory.search(query="hello")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["key"] == "greeting"

    @pytest.mark.asyncio
    async def test_search_by_key(self, memory):
        memory.save(key="meeting_notes", content="discussed Q3 budget")

        result = memory.search(query="meeting")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["key"] == "meeting_notes"

    @pytest.mark.asyncio
    async def test_search_by_tag(self, memory):
        memory.save(key="item1", content="test", tags=["important"])
        memory.save(key="item2", content="test", tags=["trivial"])

        result = memory.search(query="important")
        parsed = json.loads(result)
        assert len(parsed) == 1
        assert parsed[0]["key"] == "item1"

    @pytest.mark.asyncio
    async def test_search_no_results(self, memory):
        result = memory.search(query="nonexistent")
        assert result == "No memories found."

    @pytest.mark.asyncio
    async def test_search_limit(self, memory):
        for i in range(10):
            memory.save(key=f"item_{i}", content=f"content {i}")

        result = memory.search(query="content", limit=3)
        parsed = json.loads(result)
        assert len(parsed) == 3


class TestMemoryDelete:
    @pytest.mark.asyncio
    async def test_delete_existing(self, memory):
        memory.save(key="test", content="hello")
        result = memory.delete(key="test")
        assert "Deleted memory: test" in result

        # Verify deleted
        cursor = memory._conn.execute("SELECT * FROM memories WHERE key='test'")
        assert cursor.fetchone() is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, memory):
        result = memory.delete(key="ghost")
        assert "not found" in result


class TestMemoryListAll:
    @pytest.mark.asyncio
    async def test_list_all_empty(self, memory):
        result = memory.list_all()
        assert result == "No memories stored."

    @pytest.mark.asyncio
    async def test_list_all_returns_entries(self, memory):
        memory.save(key="a", content="alpha")
        memory.save(key="b", content="beta")

        result = memory.list_all()
        parsed = json.loads(result)
        assert len(parsed) == 2

    @pytest.mark.asyncio
    async def test_list_all_limit(self, memory):
        for i in range(10):
            memory.save(key=f"item_{i}", content=f"content {i}")

        result = memory.list_all(limit=3)
        parsed = json.loads(result)
        assert len(parsed) == 3


class TestMemoryToolIntegration:
    def test_get_tools_returns_prefixed(self):
        mem = Memory(path=":memory:")
        tools = mem.get_tools()
        names = [t.__name__ for t in tools]
        assert "memory__save" in names
        assert "memory__search" in names
        assert "memory__delete" in names
        assert "memory__list_all" in names

    def test_tools_have_schemas(self):
        mem = Memory(path=":memory:")
        tools = mem.get_tools()
        for t in tools:
            assert hasattr(t, "_tool_schema")
            schema = t._tool_schema
            func_schema = schema.get("function", schema)
            assert func_schema["name"].startswith("memory__")
