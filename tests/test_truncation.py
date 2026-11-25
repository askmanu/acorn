"""Tests for truncated output handling (finish_reason="length")."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, Field

from acorn import Module, tool
from acorn.exceptions import AcornError, ParseError, TruncatedOutputError
from acorn.llm.litellm_client import (
    _accumulate_streaming_response,
    _handle_streaming_response,
)
from tests.test_agentic_loop import MockResponse, create_tool_call


class DefaultedOutput(BaseModel):
    items: list[str] = Field(default_factory=list)


class StringOutput(BaseModel):
    result: str


class TruncatedModule(Module):
    model = "test-model"
    max_tokens = 16000
    final_output = DefaultedOutput


def with_usage(response, **usage):
    """Attach a LiteLLM-style usage object to a mock response."""
    response.usage = MagicMock()
    response.usage.model_dump.return_value = usage
    return response


async def test_truncated_finish_is_not_accepted_as_output():
    """{} from a length-truncated response must not validate into an empty result."""
    finish_call = create_tool_call("__finish__", {})
    response = MockResponse(tool_calls=[finish_call], finish_reason="length")

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=response,
    ):
        mod = TruncatedModule()
        with pytest.raises(TruncatedOutputError) as excinfo:
            await mod()

    error = excinfo.value
    assert isinstance(error, AcornError)
    assert error.token_limit == 16000
    assert error.raw_output == "{}"
    assert "finish_reason='length'" in str(error)


async def test_complete_finish_with_defaults_is_accepted():
    """An empty result is valid when the response was not truncated."""
    finish_call = create_tool_call("__finish__", {})
    response = MockResponse(tool_calls=[finish_call], finish_reason="stop")

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=response,
    ):
        result = await TruncatedModule()()

    assert result.items == []


async def test_assistant_history_keeps_finish_reason_and_usage():
    """History records audit data even when the call fails validation."""
    finish_call = create_tool_call("__finish__", {})
    response = with_usage(
        MockResponse(tool_calls=[finish_call], finish_reason="length"),
        prompt_tokens=12,
        completion_tokens=16000,
        total_tokens=16012,
    )

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=response,
    ):
        mod = TruncatedModule()
        with pytest.raises(TruncatedOutputError):
            await mod()

    assistant = [m for m in mod.history if m["role"] == "assistant"][-1]
    assert assistant["finish_reason"] == "length"
    assert assistant["usage"]["total_tokens"] == 16012


async def test_audit_keys_are_not_sent_back_to_the_provider():
    """finish_reason/usage on history messages are stripped from requests."""
    invalid = create_tool_call("__finish__", {"result": "bad"}, "call_1")
    valid = create_tool_call("__finish__", {"result": "ok"}, "call_2")

    class Output(BaseModel):
        result: int

    class M(Module):
        model = "test-model"
        final_output = Output
        max_parse_retries = 1

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion", new_callable=AsyncMock
    ) as mock_completion:
        mock_completion.side_effect = [
            with_usage(
                MockResponse(tool_calls=[invalid], finish_reason="stop"),
                total_tokens=10,
            ),
            with_usage(
                MockResponse(tool_calls=[valid], finish_reason="stop"),
                total_tokens=20,
            ),
        ]

        with pytest.raises(ParseError):
            # "bad" is not an int, and the corrected reply also fails validation.
            await M()()

        retry_messages = mock_completion.call_args_list[1].kwargs["messages"]
        assert all("finish_reason" not in msg for msg in retry_messages)
        assert all("usage" not in msg for msg in retry_messages)


async def test_truncated_finish_in_agentic_loop_is_not_accepted():
    finish_call = create_tool_call("__finish__", {})

    class Output(BaseModel):
        items: list[str] = Field(default_factory=list)

    class M(Module):
        model = "test-model"
        max_steps = 2
        final_output = Output

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=MockResponse(tool_calls=[finish_call], finish_reason="length"),
    ):
        with pytest.raises(TruncatedOutputError):
            await M()()


async def test_forced_termination_truncation_is_not_swallowed():
    """A truncated forced __finish__ raises instead of falling back to instruction."""
    @tool
    def endless_tool() -> str:
        """Keep going."""
        return "continue"

    class Output(BaseModel):
        result: str

    class M(Module):
        model = "test-model"
        max_steps = 1
        tools = [endless_tool]
        final_output = Output

    endless = MockResponse(tool_calls=[create_tool_call("endless_tool", {}, "call_1")])
    truncated = MockResponse(
        tool_calls=[create_tool_call("__finish__", {}, "call_finish")],
        finish_reason="length",
    )

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion", new_callable=AsyncMock
    ) as mock_completion:
        mock_completion.side_effect = [endless, truncated]

        with pytest.raises(TruncatedOutputError):
            await M()()

        # No fallback request to _finish_by_instruction
        assert mock_completion.call_count == 2


async def test_truncated_reply_during_parse_retry_raises():
    class Output(BaseModel):
        count: int

    class M(Module):
        model = "test-model"
        final_output = Output
        max_parse_retries = 1

    invalid = create_tool_call("__finish__", {"count": "bad"}, "call_1")
    truncated = create_tool_call("__finish__", {"count": 1}, "call_2")

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion", new_callable=AsyncMock
    ) as mock_completion:
        mock_completion.side_effect = [
            MockResponse(tool_calls=[invalid], finish_reason="stop"),
            MockResponse(tool_calls=[truncated], finish_reason="length"),
        ]

        mod = M()
        with pytest.raises(TruncatedOutputError):
            await mod()

        # The retry exchange is in history, including the truncated reply.
        assert [m["role"] for m in mod.history] == [
            "system", "user", "assistant", "tool", "assistant"
        ]
        assert mod.history[-1]["finish_reason"] == "length"


async def test_truncated_finish_by_instruction_raises():
    @tool
    def endless_tool() -> str:
        """Keep going."""
        return "continue"

    class Output(BaseModel):
        result: str

    class M(Module):
        model = "test-model"
        max_steps = 1
        tools = [endless_tool]
        final_output = Output

    endless = MockResponse(tool_calls=[create_tool_call("endless_tool", {}, "call_1")])
    truncated = MockResponse(
        tool_calls=[create_tool_call("__finish__", {}, "call_finish")],
        finish_reason="length",
    )

    async def llm(**kwargs):
        if "tool_choice" in kwargs:
            raise Exception("tool_choice not supported")
        return next(replies)

    replies = iter([endless, truncated])

    with patch("acorn.llm.litellm_client.litellm.acompletion", new=AsyncMock(side_effect=llm)):
        mod = M()
        with pytest.raises(TruncatedOutputError):
            await mod()

        # The instruction and the failed __finish__ call are in the trace.
        assert any("step limit" in str(msg.get("content")) for msg in mod.history)
        assert mod.history[-1]["finish_reason"] == "length"


async def test_truncated_branch_keeps_typed_error():
    class Branch(Module):
        """Branch with defaulted output."""

        model = "test-model"
        final_output = DefaultedOutput

    class Parent(Module):
        model = "test-model"
        branches = [Branch]
        final_output = StringOutput

    truncated = MockResponse(
        tool_calls=[create_tool_call("__finish__", {}, "call_finish")],
        finish_reason="length",
    )

    with patch(
        "acorn.llm.litellm_client.litellm.acompletion",
        new_callable=AsyncMock,
        return_value=truncated,
    ):
        parent = Parent()
        parent.history = [
            {"role": "system", "content": "Parent"},
            {"role": "user", "content": "test"},
        ]
        with pytest.raises(TruncatedOutputError):
            await parent._execute_branch(Branch, "end_result")


async def test_streaming_accumulates_finish_reason_and_usage():
    async def stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].finish_reason = None
        chunk.choices[0].delta = MagicMock(content="partial", reasoning_content=None, tool_calls=[])
        chunk.usage = None
        yield chunk

        final = MagicMock()
        final.choices = [MagicMock()]
        final.choices[0].finish_reason = "length"
        final.choices[0].delta = None
        final.usage = MagicMock()
        final.usage.model_dump.return_value = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
        yield final

    result = await _accumulate_streaming_response(stream())

    assert result["finish_reason"] == "length"
    assert result["usage"]["total_tokens"] == 3


async def test_streaming_with_callback_accumulates_usage():
    async def stream():
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].finish_reason = "length"
        chunk.choices[0].delta = MagicMock(content="text", reasoning_content=None, tool_calls=[])
        chunk.usage = MagicMock()
        chunk.usage.model_dump.return_value = {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        }
        yield chunk

    result = await _handle_streaming_response(stream(), lambda chunk: None)

    assert result["finish_reason"] == "length"
    assert result["usage"]["total_tokens"] == 3
