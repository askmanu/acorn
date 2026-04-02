"""Tests for the Service base class and Module integration."""

import pytest
import json
from unittest.mock import patch, MagicMock, AsyncMock
from pydantic import BaseModel

from acorn import Module, Service, tool
from acorn.exceptions import ServiceError, ToolConflictError
from acorn.service import _to_snake_case, _prefix_tool


# --- Test helpers ---

class MockResponse:
    """Mock LiteLLM response object."""
    def __init__(self, content=None, tool_calls=None, finish_reason="stop"):
        self.choices = [MagicMock()]
        message = MagicMock()
        message.content = content
        message.tool_calls = tool_calls or []
        self.choices[0].message = message
        self.choices[0].finish_reason = finish_reason


def create_tool_call(name: str, arguments: dict, call_id: str = "call_123"):
    """Create a mock tool call."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


# --- Test services ---

class EmailService(Service):
    """Send and search emails."""

    def __init__(self, token: str = "test-token"):
        self.token = token
        self.setup_called = False
        self.teardown_called = False

    async def setup(self):
        self.setup_called = True

    async def teardown(self):
        self.teardown_called = True

    async def health(self) -> bool:
        return self.setup_called

    @tool
    def send(self, to: str, subject: str, body: str) -> str:
        """Send an email.

        Args:
            to: Recipient address
            subject: Email subject
            body: Email body
        """
        return f"Sent to {to}: {subject}"

    @tool
    def search(self, query: str) -> str:
        """Search emails.

        Args:
            query: Search query
        """
        return f"Results for: {query}"


class CalendarService(Service):
    """Manage calendar events."""

    def __init__(self):
        pass

    @tool
    def create_event(self, title: str, date: str) -> str:
        """Create a calendar event.

        Args:
            title: Event title
            date: Event date
        """
        return f"Created: {title} on {date}"

    @tool
    def search(self, query: str) -> str:
        """Search calendar events.

        Args:
            query: Search query
        """
        return f"Calendar results: {query}"


class EmptyService(Service):
    """A service with no tools."""
    pass


class Output(BaseModel):
    result: str


# =============================================================================
# Test: _to_snake_case
# =============================================================================

class TestToSnakeCase:
    def test_simple_name(self):
        assert _to_snake_case("Gmail") == "gmail"

    def test_camel_case(self):
        assert _to_snake_case("GoogleCalendar") == "google_calendar"

    def test_acronym_prefix(self):
        assert _to_snake_case("E2BSandbox") == "e2b_sandbox"

    def test_already_lowercase(self):
        assert _to_snake_case("memory") == "memory"

    def test_all_caps(self):
        assert _to_snake_case("API") == "api"

    def test_mixed_acronym(self):
        assert _to_snake_case("HTTPClient") == "http_client"

    def test_single_char(self):
        assert _to_snake_case("A") == "a"


# =============================================================================
# Test: _prefix_tool
# =============================================================================

class TestPrefixTool:
    def test_prefix_name(self):
        @tool
        def send(to: str) -> str:
            """Send something."""
            return to

        prefixed = _prefix_tool(send, "gmail")
        assert prefixed.__name__ == "gmail__send"

    def test_prefix_schema_updated(self):
        @tool
        def send(to: str) -> str:
            """Send something."""
            return to

        prefixed = _prefix_tool(send, "gmail")
        schema = prefixed._tool_schema
        func_schema = schema.get("function", schema)
        assert func_schema["name"] == "gmail__send"

    def test_prefix_preserves_behavior(self):
        @tool
        def add(a: int, b: int) -> int:
            """Add two numbers."""
            return a + b

        prefixed = _prefix_tool(add, "math")
        assert prefixed(3, 4) == 7

    def test_original_unchanged(self):
        @tool
        def send(to: str) -> str:
            """Send something."""
            return to

        _prefix_tool(send, "gmail")
        # Original tool unchanged
        assert send.__name__ == "send"
        assert send._tool_schema["function"]["name"] == "send"


# =============================================================================
# Test: Service base class
# =============================================================================

class TestServiceBase:
    def test_name_from_class(self):
        svc = EmailService()
        assert svc.name == "EmailService"

    def test_description_from_docstring(self):
        svc = EmailService()
        assert svc.description == "Send and search emails."

    def test_description_empty_if_no_docstring(self):
        class Bare(Service):
            pass
        svc = Bare()
        assert svc.description == ""

    def test_get_tools_returns_prefixed(self):
        svc = EmailService()
        tools = svc.get_tools()
        names = [t.__name__ for t in tools]
        assert "email_service__send" in names
        assert "email_service__search" in names

    def test_get_tools_have_schema(self):
        svc = EmailService()
        tools = svc.get_tools()
        for t in tools:
            assert hasattr(t, "_tool_schema")

    def test_get_tools_callable(self):
        svc = EmailService()
        tools = svc.get_tools()
        send_tool = next(t for t in tools if "send" in t.__name__)
        result = send_tool(to="bob@test.com", subject="Hi", body="Hello")
        assert "Sent to bob@test.com" in result

    def test_empty_service_returns_no_tools(self):
        svc = EmptyService()
        assert svc.get_tools() == []

    @pytest.mark.asyncio
    async def test_default_lifecycle(self):
        svc = Service()
        await svc.setup()
        assert await svc.health() is True
        await svc.teardown()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        svc = EmailService()
        async with svc:
            assert svc.setup_called is True
        assert svc.teardown_called is True


# =============================================================================
# Test: Module integration with Service
# =============================================================================

class TestModuleServiceIntegration:
    def test_service_expands_in_tools(self):
        email = EmailService()

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = [email]

        agent = Agent()
        tool_names = [t.__name__ for t in agent._collected_tools]
        assert "email_service__send" in tool_names
        assert "email_service__search" in tool_names

    def test_cherry_pick_no_prefix(self):
        email = EmailService()

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = [email.send]

        agent = Agent()
        tool_names = [t.__name__ for t in agent._collected_tools]
        assert "send" in tool_names

    def test_mixed_tools_list(self):
        @tool
        def calculate(expr: str) -> str:
            """Calculate expression."""
            return expr

        email = EmailService()

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = [calculate, email, email.search]

        # This should raise ToolConflictError because email.search (unprefixed)
        # conflicts with email_service__search from the expanded service.
        # Actually no — "search" != "email_service__search", no conflict.
        agent = Agent()
        tool_names = [t.__name__ for t in agent._collected_tools]
        assert "calculate" in tool_names
        assert "email_service__send" in tool_names
        assert "email_service__search" in tool_names
        assert "search" in tool_names

    def test_two_services_no_conflict(self):
        """Two services with same method names don't conflict due to prefixing."""
        email = EmailService()
        calendar = CalendarService()

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = [email, calendar]

        agent = Agent()
        tool_names = [t.__name__ for t in agent._collected_tools]
        assert "email_service__search" in tool_names
        assert "calendar_service__search" in tool_names

    def test_services_tracked(self):
        email = EmailService()
        calendar = CalendarService()

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = [email, calendar]

        agent = Agent()
        assert email in agent._services
        assert calendar in agent._services

    @pytest.mark.asyncio
    async def test_lifecycle_called(self):
        """setup() and teardown() are called during module execution."""
        email = EmailService()

        class Agent(Module):
            model = "test-model"
            final_output = Output

        agent = Agent()
        agent._services = [email]
        agent.tools = [email]
        agent._collected_tools = email.get_tools()

        finish_call = create_tool_call(
            "__finish__", {"result": "done"}, call_id="call_1"
        )
        mock_response = {
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "__finish__", "arguments": '{"result": "done"}'},
                }
            ],
            "reasoning_content": None,
        }

        with patch("acorn.module.call_llm", new_callable=AsyncMock, return_value=mock_response):
            result = await agent(result="test")

        assert email.setup_called is True
        assert email.teardown_called is True

    @pytest.mark.asyncio
    async def test_setup_failure_raises_service_error(self):
        class FailingService(Service):
            """Fails on setup."""
            async def setup(self):
                raise ConnectionError("Cannot connect")

        svc = FailingService()

        class Agent(Module):
            model = "test-model"
            final_output = Output

        agent = Agent()
        agent._services = [svc]

        with pytest.raises(ServiceError, match="setup failed"):
            await agent(result="test")

    @pytest.mark.asyncio
    async def test_teardown_called_on_error(self):
        """teardown() is called even if execution fails."""
        email = EmailService()

        class Agent(Module):
            model = "test-model"
            final_output = Output

        agent = Agent()
        agent._services = [email]

        with patch("acorn.module.call_llm", new_callable=AsyncMock, side_effect=Exception("LLM failed")):
            with pytest.raises(Exception, match="LLM failed"):
                await agent(result="test")

        assert email.teardown_called is True

    @pytest.mark.asyncio
    async def test_health_check(self):
        email = EmailService()
        assert await email.health() is False
        await email.setup()
        assert await email.health() is True
        await email.teardown()


# =============================================================================
# Test: Service tool execution in agentic loop
# =============================================================================

class TestServiceToolExecution:
    @pytest.mark.asyncio
    async def test_service_tool_callable_in_loop(self):
        """Service tools can be called by the LLM via prefixed names."""
        email = EmailService()

        class Agent(Module):
            model = "test-model"
            max_steps = 3
            tools = [email]
            final_output = Output

        agent = Agent()

        # Step 1: LLM calls email_service__send
        response1 = {
            "content": "I'll send an email.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {
                        "name": "email_service__send",
                        "arguments": json.dumps({"to": "bob@test.com", "subject": "Hi", "body": "Hello"}),
                    },
                }
            ],
            "reasoning_content": None,
        }

        # Step 2: LLM calls __finish__
        response2 = {
            "content": "Done.",
            "tool_calls": [
                {
                    "id": "call_2",
                    "function": {
                        "name": "__finish__",
                        "arguments": json.dumps({"result": "Email sent successfully"}),
                    },
                }
            ],
            "reasoning_content": None,
        }

        with patch("acorn.module.call_llm", new_callable=AsyncMock, side_effect=[response1, response2]):
            result = await agent()

        assert result.result == "Email sent successfully"
