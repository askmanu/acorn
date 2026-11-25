"""Behaviour needed by current models: no default temperature and finishing without a forced
tool_choice (Claude Sonnet/Opus 5.5 reject both a non-default temperature and a forced
tool_choice)."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from acorn import Module, tool
from acorn.exceptions import AcornError
from tests.test_agentic_loop import MockResponse, create_tool_call


class Output(BaseModel):
    result: str


def finish(result: str = "done", call_id: str = "call_finish"):
    return MockResponse(tool_calls=[create_tool_call("__finish__", {"result": result}, call_id)])


async def test_temperature_not_sent_by_default():
    class M(Module):
        model = "test-model"
        final_output = Output

    with patch("acorn.llm.litellm_client.litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = finish()
        await M()()
        assert "temperature" not in llm.call_args.kwargs


async def test_temperature_sent_when_set():
    class M(Module):
        model = "test-model"
        final_output = Output
        temperature = 0.2

    with patch("acorn.llm.litellm_client.litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = finish()
        await M()()
        assert llm.call_args.kwargs["temperature"] == 0.2


async def test_docstring_is_not_used_as_system_prompt():
    class M(Module):
        """Human-facing documentation, not a prompt."""

        model = "test-model"
        final_output = Output

    with patch("acorn.llm.litellm_client.litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = finish()
        await M()()
        assert llm.call_args.kwargs["messages"][0] == {"role": "system", "content": ""}


async def test_system_prompt_is_used():
    class M(Module):
        """Human-facing documentation, not a prompt."""

        system_prompt = "Explicit."
        model = "test-model"
        final_output = Output

    with patch("acorn.llm.litellm_client.litellm.acompletion", new_callable=AsyncMock) as llm:
        llm.return_value = finish()
        await M()()
        assert llm.call_args.kwargs["messages"][0]["content"] == "Explicit."


def endless_module():
    @tool
    def endless_tool() -> str:
        """A tool that keeps running."""
        return "continue"

    class Endless(Module):
        model = "test-model"
        max_steps = 1
        tools = [endless_tool]
        final_output = Output

    return Endless()


async def test_max_steps_finishes_by_instruction_when_forced_choice_is_rejected():
    endless = MockResponse(tool_calls=[create_tool_call("endless_tool", {}, "call_1")])

    async def llm(**kwargs):
        if "tool_choice" in kwargs:
            raise Exception("does not support forced tool use")
        last = kwargs["messages"][-1]
        if last["role"] == "user" and "step limit" in str(last["content"]):
            return finish("by instruction")
        return endless

    acompletion = "acorn.llm.litellm_client.litellm.acompletion"
    with patch(acompletion, new=AsyncMock(side_effect=llm)) as mock:
        result = await endless_module()()
        assert result.result == "by instruction"
        last = mock.call_args_list[-1].kwargs
        assert "tool_choice" not in last


async def test_instruction_answers_other_tool_calls_without_running_them():
    endless = MockResponse(tool_calls=[create_tool_call("endless_tool", {}, "call_1")])
    replies = iter([endless, endless, finish("second try")])
    sent: list[list[dict]] = []

    async def llm(**kwargs):
        if "tool_choice" in kwargs:
            raise Exception("does not support forced tool use")
        sent.append(list(kwargs["messages"]))  # the history list grows after the call
        return next(replies)

    with patch("acorn.llm.litellm_client.litellm.acompletion", new=AsyncMock(side_effect=llm)):
        mod = endless_module()
        result = await mod()
        assert result.result == "second try"
        assert sent[-1][-1] == {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "Error: step limit reached; this tool was not run. Call __finish__ now.",
        }
        # The fallback exchange is part of history: the trace matches what the
        # model saw and returned.
        assert any("step limit" in str(msg.get("content")) for msg in mod.history)
        assert mod.history[-1]["tool_calls"][0]["function"]["name"] == "__finish__"


async def test_max_steps_instruction_gives_up_after_retries():
    endless = MockResponse(tool_calls=[create_tool_call("endless_tool", {}, "call_1")])

    async def llm(**kwargs):
        if "tool_choice" in kwargs:
            raise Exception("does not support forced tool use")
        return endless

    acompletion = "acorn.llm.litellm_client.litellm.acompletion"
    with patch(acompletion, new=AsyncMock(side_effect=llm)) as mock:
        mod = endless_module()
        mod.max_parse_retries = 0

        with pytest.raises(AcornError, match="did not call __finish__"):
            await mod()

        # Step 1, the forced attempt, and one instruction attempt
        assert mock.call_count == 3
        # The instruction and the rejected tool call are kept in history.
        assert any("step limit" in str(msg.get("content")) for msg in mod.history)
        assert any(msg.get("role") == "tool" for msg in mod.history)
