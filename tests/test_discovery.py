"""Tests for the tool discovery system."""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import BaseModel

from acorn import Module, Service, tool
from acorn.discovery import ToolRegistry, _tokenize


# --- Test tools ---

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        to: Email address of the recipient
        subject: Subject line
        body: Email body content
    """
    return f"Sent to {to}"


@tool
def search_emails(query: str, limit: int = 10) -> str:
    """Search through emails by keyword.

    Args:
        query: Search query
        limit: Max results
    """
    return f"Found emails matching: {query}"


@tool
def create_event(title: str, date: str) -> str:
    """Create a calendar event.

    Args:
        title: Event title
        date: Event date
    """
    return f"Created: {title}"


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: Math expression to evaluate
    """
    return str(eval(expression))


@tool
def web_search(query: str) -> str:
    """Search the web for information.

    Args:
        query: Search query
    """
    return f"Web results for: {query}"


ALL_TOOLS = [send_email, search_emails, create_event, calculate, web_search]


class Output(BaseModel):
    result: str


# =============================================================================
# Test: _tokenize
# =============================================================================

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("send email")
        assert "send" in tokens
        assert "email" in tokens

    def test_underscore_split(self):
        tokens = _tokenize("send_email")
        assert "send" in tokens
        assert "email" in tokens

    def test_short_tokens_filtered(self):
        tokens = _tokenize("a is the")
        assert "a" not in tokens
        assert "is" in tokens
        assert "the" in tokens

    def test_case_insensitive(self):
        tokens = _tokenize("Send EMAIL")
        assert "send" in tokens
        assert "email" in tokens


# =============================================================================
# Test: ToolRegistry
# =============================================================================

class TestToolRegistry:
    def test_init_indexes_tools(self):
        registry = ToolRegistry(ALL_TOOLS)
        assert len(registry) == len(ALL_TOOLS)

    def test_search_by_keyword(self):
        registry = ToolRegistry(ALL_TOOLS)
        results = registry.search("email")
        names = [r.get("function", r)["name"] for r in results]
        assert "send_email" in names
        assert "search_emails" in names

    def test_search_returns_schemas(self):
        registry = ToolRegistry(ALL_TOOLS)
        results = registry.search("email")
        for r in results:
            func = r.get("function", r)
            assert "name" in func
            assert "parameters" in func

    def test_search_limit(self):
        registry = ToolRegistry(ALL_TOOLS)
        results = registry.search("search", limit=1)
        assert len(results) == 1

    def test_search_no_results(self):
        registry = ToolRegistry(ALL_TOOLS)
        results = registry.search("zzzznonexistent")
        assert results == []

    def test_search_empty_query(self):
        registry = ToolRegistry(ALL_TOOLS)
        results = registry.search("")
        assert results == []

    def test_search_name_weighted(self):
        """Tools matching by name score higher than by description only."""
        registry = ToolRegistry(ALL_TOOLS)
        results = registry.search("calculate")
        names = [r.get("function", r)["name"] for r in results]
        assert names[0] == "calculate"

    def test_get_exact_match(self):
        registry = ToolRegistry(ALL_TOOLS)
        result = registry.get("send_email")
        assert result is not None
        func = result.get("function", result)
        assert func["name"] == "send_email"

    def test_get_not_found(self):
        registry = ToolRegistry(ALL_TOOLS)
        assert registry.get("nonexistent") is None

    def test_add_tool(self):
        registry = ToolRegistry([send_email])
        assert len(registry) == 1

        registry.add(calculate)
        assert len(registry) == 2
        assert registry.get("calculate") is not None

    def test_remove_tool(self):
        registry = ToolRegistry(ALL_TOOLS)
        initial_len = len(registry)

        registry.remove("send_email")
        assert len(registry) == initial_len - 1
        assert registry.get("send_email") is None

    def test_remove_nonexistent(self):
        registry = ToolRegistry(ALL_TOOLS)
        registry.remove("nonexistent")  # Should not raise

    def test_list_all(self):
        registry = ToolRegistry(ALL_TOOLS)
        listing = registry.list_all()
        assert len(listing) == len(ALL_TOOLS)
        for item in listing:
            assert "name" in item
            assert "description" in item

    def test_skips_internal_tools(self):
        """search() skips tools starting with __."""
        @tool
        def __finish__(**kwargs) -> str:
            """Finish the task."""
            return "done"

        registry = ToolRegistry([send_email, __finish__])
        results = registry.search("finish")
        # __finish__ should not appear in search results
        names = [r.get("function", r)["name"] for r in results]
        assert "__finish__" not in names

    def test_works_with_service_prefixed_tools(self):
        """Search finds service-prefixed tools by their base name."""

        class EmailSvc(Service):
            """Email service."""
            @tool
            def send(self, to: str) -> str:
                """Send an email."""
                return f"sent to {to}"

        svc = EmailSvc()
        tools = svc.get_tools()
        registry = ToolRegistry(tools)

        # Search for "send" should find "email_svc__send"
        results = registry.search("send")
        assert len(results) >= 1
        names = [r.get("function", r)["name"] for r in results]
        assert "email_svc__send" in names

        # Search for "email" should also match (from tool name prefix)
        results = registry.search("email")
        assert len(results) >= 1


# =============================================================================
# Test: Module with tool_discovery
# =============================================================================

class TestModuleDiscovery:
    def test_default_is_none(self):
        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = ALL_TOOLS

        agent = Agent()
        assert agent.tool_discovery is None

    def test_discovery_attribute_accepted(self):
        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = ALL_TOOLS
            tool_discovery = "search"

        agent = Agent()
        assert agent.tool_discovery == "search"

    @pytest.mark.asyncio
    async def test_discovery_search_only_visible_tools(self):
        """With tool_discovery='search', LLM sees only search_tools + __finish__."""

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = ALL_TOOLS
            final_output = Output
            tool_discovery = "search"

        agent = Agent()

        captured_tools = []

        async def mock_call_llm(messages, model, tools, **kwargs):
            captured_tools.extend(tools)
            return {
                "content": "Done",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "__finish__",
                            "arguments": json.dumps({"result": "done"}),
                        },
                    }
                ],
                "reasoning_content": None,
            }

        with patch("acorn.module.call_llm", side_effect=mock_call_llm):
            await agent()

        # Only search_tools and __finish__ should be in the schemas sent to LLM
        tool_names = [t["function"]["name"] for t in captured_tools]
        assert "search_tools" in tool_names
        assert "__finish__" in tool_names
        assert len(tool_names) == 2
        # The actual tools should NOT be in the schemas
        assert "send_email" not in tool_names
        assert "calculate" not in tool_names

    @pytest.mark.asyncio
    async def test_discovery_tools_still_callable(self):
        """Even with discovery, all tools remain callable by the LLM."""

        class Agent(Module):
            model = "test-model"
            max_steps = 5
            tools = ALL_TOOLS
            final_output = Output
            tool_discovery = "search"

        agent = Agent()

        call_count = 0

        async def mock_call_llm(messages, model, tools, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Step 1: LLM searches for tools
                return {
                    "content": "Let me search for tools.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "search_tools",
                                "arguments": json.dumps({"query": "calculate math"}),
                            },
                        }
                    ],
                    "reasoning_content": None,
                }
            elif call_count == 2:
                # Step 2: LLM calls the discovered tool
                return {
                    "content": "Now I'll calculate.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {
                                "name": "calculate",
                                "arguments": json.dumps({"expression": "2+2"}),
                            },
                        }
                    ],
                    "reasoning_content": None,
                }
            else:
                # Step 3: Finish
                return {
                    "content": "Done.",
                    "tool_calls": [
                        {
                            "id": "call_3",
                            "function": {
                                "name": "__finish__",
                                "arguments": json.dumps({"result": "4"}),
                            },
                        }
                    ],
                    "reasoning_content": None,
                }

        with patch("acorn.module.call_llm", side_effect=mock_call_llm):
            result = await agent()

        assert result.result == "4"

    @pytest.mark.asyncio
    async def test_no_discovery_sends_all_tools(self):
        """Without discovery, all tool schemas are sent to LLM."""

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = ALL_TOOLS
            final_output = Output
            # tool_discovery defaults to None

        agent = Agent()

        captured_tools = []

        async def mock_call_llm(messages, model, tools, **kwargs):
            captured_tools.extend(tools)
            return {
                "content": "Done",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {
                            "name": "__finish__",
                            "arguments": json.dumps({"result": "done"}),
                        },
                    }
                ],
                "reasoning_content": None,
            }

        with patch("acorn.module.call_llm", side_effect=mock_call_llm):
            await agent()

        tool_names = [t["function"]["name"] for t in captured_tools]
        assert "send_email" in tool_names
        assert "calculate" in tool_names
        assert "__finish__" in tool_names
        assert "search_tools" not in tool_names

    @pytest.mark.asyncio
    async def test_search_tool_returns_results(self):
        """The search_tools tool returns matching tool schemas as JSON."""

        class Agent(Module):
            model = "test-model"
            max_steps = 5
            tools = ALL_TOOLS
            final_output = Output
            tool_discovery = "search"

        agent = Agent()

        search_result = None

        async def mock_call_llm(messages, model, tools, **kwargs):
            nonlocal search_result
            # Check if previous messages contain a tool result from search_tools
            for msg in messages:
                if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_1":
                    search_result = msg.get("content")

            if search_result is None:
                return {
                    "content": "Searching.",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {
                                "name": "search_tools",
                                "arguments": json.dumps({"query": "email"}),
                            },
                        }
                    ],
                    "reasoning_content": None,
                }
            else:
                return {
                    "content": "Done.",
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "function": {
                                "name": "__finish__",
                                "arguments": json.dumps({"result": "found tools"}),
                            },
                        }
                    ],
                    "reasoning_content": None,
                }

        with patch("acorn.module.call_llm", side_effect=mock_call_llm):
            await agent()

        assert search_result is not None
        parsed = json.loads(search_result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0
        names = [t["name"] for t in parsed]
        assert "send_email" in names or "search_emails" in names


# =============================================================================
# Test: _generate_search_tool
# =============================================================================

class TestGenerateSearchTool:
    def test_generates_callable_with_schema(self):
        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = ALL_TOOLS

        agent = Agent()
        registry = ToolRegistry(ALL_TOOLS)
        search_tool = agent._generate_search_tool(registry)

        assert hasattr(search_tool, "_tool_schema")
        assert search_tool._tool_schema["function"]["name"] == "search_tools"

    def test_search_tool_returns_json(self):
        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = ALL_TOOLS

        agent = Agent()
        registry = ToolRegistry(ALL_TOOLS)
        search_tool = agent._generate_search_tool(registry)

        result = search_tool(query="email")
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) > 0

    def test_search_tool_no_results(self):
        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = ALL_TOOLS

        agent = Agent()
        registry = ToolRegistry(ALL_TOOLS)
        search_tool = agent._generate_search_tool(registry)

        result = search_tool(query="zzzznonexistent")
        assert "No tools found" in result
