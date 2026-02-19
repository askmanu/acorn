"""Tests for the branching system (Phase 9)."""

import pytest
import json
from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from acorn import Module, tool
from acorn.exceptions import BranchError, ToolConflictError


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


# --- Shared models ---

class ClaimInput(BaseModel):
    claim: str

class VerificationOutput(BaseModel):
    verified: bool
    explanation: str

class AnswerOutput(BaseModel):
    answer: str


# --- Branch module definitions ---

class FactCheckBranch(Module):
    """Verify factual claims using available tools."""
    model = "test-model"
    system_prompt = "Verify factual claims using available tools."
    initial_input = ClaimInput
    final_output = VerificationOutput


class MultiTurnBranch(Module):
    """A branch that uses multiple steps."""
    model = "test-model"
    initial_input = ClaimInput
    final_output = VerificationOutput
    max_steps = 3


class NoBranchInput(Module):
    """A branch with no initial_input."""
    model = "test-model"
    final_output = AnswerOutput


# =============================================================================
# Test: Branch validation
# =============================================================================

class TestBranchValidation:
    def test_branches_must_be_list(self):
        class Bad(Module):
            model = "test-model"
            branches = {"not": "a list"}
            final_output = AnswerOutput
        with pytest.raises(ValueError, match="branches must be a list"):
            Bad()

    def test_branch_values_must_be_module_subclass(self):
        class Bad(Module):
            model = "test-model"
            branches = ["not a class"]
            final_output = AnswerOutput
        with pytest.raises(ValueError, match="must be a Module subclass"):
            Bad()

    def test_branch_values_must_be_subclass_not_instance(self):
        class Bad(Module):
            model = "test-model"
            branches = [int]
            final_output = AnswerOutput
        with pytest.raises(ValueError, match="must be a Module subclass"):
            Bad()

    def test_duplicate_branch_names_rejected(self):
        class BranchA(Module):
            model = "test-model"
            final_output = AnswerOutput

        class BranchA(Module):  # Same name
            model = "test-model"
            final_output = VerificationOutput

        class Bad(Module):
            model = "test-model"
            branches = [BranchA, BranchA]
            final_output = AnswerOutput
        with pytest.raises(ValueError, match="Duplicate branch class name"):
            Bad()


    def test_empty_branches_is_valid(self):
        class Good(Module):
            model = "test-model"
            branches = []
            final_output = AnswerOutput
        mod = Good()
        assert mod.branches == []

    def test_valid_branches_config(self):
        class Good(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
        mod = Good()
        assert FactCheckBranch in mod.branches


# =============================================================================
# Test: branch() tool generation and listing
# =============================================================================

class TestBranchToolGeneration:
    def test_branch_tool_added_to_collected_tools(self):
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
        mod = Parent()
        tool_names = [t.__name__ for t in mod._collected_tools]
        assert "branch" in tool_names

    def test_branch_tool_not_added_without_branches(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
        mod = Parent()
        tool_names = [t.__name__ for t in mod._collected_tools]
        assert "branch" not in tool_names

    def test_branch_tool_list_mode(self):
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
        mod = Parent()

        # Find the branch tool
        branch_tool = None
        for t in mod._collected_tools:
            if t.__name__ == "branch":
                branch_tool = t
                break

        assert branch_tool is not None

        # Call with no args -> list branches (returns XML now)
        result = branch_tool()
        assert "FactCheckBranch" in result
        assert "Verify factual claims" in result
        assert "input_schema" in result

    def test_branch_tool_schema(self):
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
        mod = Parent()

        branch_tool = [t for t in mod._collected_tools if t.__name__ == "branch"][0]
        schema = branch_tool._tool_schema
        assert schema["function"]["name"] == "branch"
        assert "name" in schema["function"]["parameters"]["properties"]
        assert "merge" in schema["function"]["parameters"]["properties"]
        assert schema["function"]["parameters"]["additionalProperties"] is True

    def test_branch_tool_invalid_name(self):
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
        mod = Parent()

        branch_tool = [t for t in mod._collected_tools if t.__name__ == "branch"][0]
        with pytest.raises(BranchError, match="not found"):
            branch_tool(name="nonexistent")

    def test_branch_tool_invalid_merge_strategy(self):
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
        mod = Parent()

        branch_tool = [t for t in mod._collected_tools if t.__name__ == "branch"][0]
        with pytest.raises(BranchError, match="Invalid merge strategy"):
            branch_tool(name="FactCheckBranch", merge="invalid")


# =============================================================================
# Test: Branch execution (end_result merge)
# =============================================================================

class TestBranchExecution:
    def test_branch_execution_end_result(self):
        """Test declarative branch execution with end_result merge."""
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
            max_steps = 3

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            # Parent step 1: calls branch tool
            branch_call = create_tool_call(
                "branch",
                {"name": "FactCheckBranch", "claim": "The sky is blue"},
                "call_branch"
            )
            # Branch: calls __finish__
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "Confirmed"},
                "call_branch_finish"
            )
            # Parent step 2: calls __finish__ after getting branch result
            parent_finish = create_tool_call(
                "__finish__",
                {"answer": "Fact checked successfully"},
                "call_parent_finish"
            )

            mock_completion.side_effect = [
                MockResponse(tool_calls=[branch_call]),   # Parent step 1
                MockResponse(tool_calls=[branch_finish]),  # Branch execution
                MockResponse(tool_calls=[parent_finish]),  # Parent step 2
            ]

            mod = Parent()
            result = mod()

            assert result.answer == "Fact checked successfully"
            assert mock_completion.call_count == 3

    def test_branch_execution_passes_kwargs(self):
        """Test that branch kwargs are passed to the branch module."""
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "Sky is blue confirmed"},
                "call_branch_finish"
            )

            mock_completion.return_value = MockResponse(tool_calls=[branch_finish])

            mod = Parent()
            # Execute branch directly
            result, content = mod._execute_branch(FactCheckBranch, "end_result", claim="The sky is blue")

            assert result.verified is True
            assert result.explanation == "Sky is blue confirmed"
            # Content should be XML of the result
            assert "<verified>true</verified>" in content
            assert "<explanation>Sky is blue confirmed</explanation>" in content

    def test_branch_inherits_parent_history(self):
        """Test that branch receives parent history context."""
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
            max_steps = 3

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_call = create_tool_call(
                "branch",
                {"name": "FactCheckBranch", "claim": "Test claim"},
                "call_branch"
            )
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "OK"},
                "call_branch_finish"
            )
            parent_finish = create_tool_call(
                "__finish__",
                {"answer": "Done"},
                "call_parent_finish"
            )

            # Track messages passed to each LLM call
            call_messages = []

            def capture_calls(*args, **kwargs):
                call_messages.append(kwargs.get("messages", []))
                if len(call_messages) == 1:
                    return MockResponse(tool_calls=[branch_call])
                elif len(call_messages) == 2:
                    return MockResponse(tool_calls=[branch_finish])
                else:
                    return MockResponse(tool_calls=[parent_finish])

            mock_completion.side_effect = capture_calls

            mod = Parent()
            mod()

            # The branch call (2nd) should have inherited history
            branch_messages = call_messages[1]
            # Should have system (branch's own) + inherited user message + branch input
            roles = [m.get("role") for m in branch_messages]
            assert roles[0] == "system"  # Branch's own system prompt
            assert "user" in roles  # At least one user message (inherited + branch input)

    def test_branch_gets_own_system_prompt(self):
        """Test that branch uses its own system prompt, not parent's."""
        class Parent(Module):
            """Parent system prompt."""
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "OK"},
                "call_branch_finish"
            )
            mock_completion.return_value = MockResponse(tool_calls=[branch_finish])

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent system prompt."},
                {"role": "user", "content": "test"}
            ]

            result, _ = mod._execute_branch(FactCheckBranch, "end_result", claim="test")

            # Check that the branch's LLM call used the branch's system prompt
            call_kwargs = mock_completion.call_args.kwargs
            messages = call_kwargs["messages"]
            assert messages[0]["role"] == "system"
            assert "Verify factual claims" in messages[0]["content"]


# =============================================================================
# Test: Merge strategies
# =============================================================================

class TestMergeStrategies:
    def test_merge_end_result(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
        mod = Parent()
        result = VerificationOutput(verified=True, explanation="OK")
        content = mod._merge_end_result(result)
        assert "<verified>true</verified>" in content
        assert "<explanation>OK</explanation>" in content

    def test_merge_end_result_none(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
        mod = Parent()
        content = mod._merge_end_result(None)
        assert "<status>completed</status>" in content

    def test_merge_summarize(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
        mod = Parent()

        branch_instance = MagicMock()
        branch_instance.model = "anthropic/claude-sonnet-4-5-20250514"
        branch_instance.max_tokens = 4096
        branch_instance.history = [
            {"role": "system", "content": "Branch prompt"},
            {"role": "user", "content": "Check claim"},
        ]

        result = VerificationOutput(verified=True, explanation="Confirmed")

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            mock_completion.return_value = MockResponse(
                content="The branch verified the claim was true."
            )

            content = mod._merge_summarize(branch_instance, result)

            assert "Branch summary:" in content
            assert "The branch verified" in content
            assert "Final result:" in content
            assert mock_completion.call_count == 1


# =============================================================================
# Test: call_parent_tool()
# =============================================================================

class TestCallParentTool:
    def test_call_parent_tool_list_mode(self):
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        class Parent(Module):
            model = "test-model"
            tools = [search]
            branches = [FactCheckBranch]
            final_output = AnswerOutput

        mod = Parent()

        # Generate call_parent_tool
        call_parent_tool_func = mod._generate_parent_tool(mod._collected_tools)

        # List mode
        result = json.loads(call_parent_tool_func())
        tool_names = [t["name"] for t in result]
        assert "search" in tool_names
        assert "__finish__" not in tool_names
        assert "branch" not in tool_names

    def test_call_parent_tool_execution(self):
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        class Parent(Module):
            model = "test-model"
            tools = [search]
            branches = [FactCheckBranch]
            final_output = AnswerOutput

        mod = Parent()
        call_parent_tool_func = mod._generate_parent_tool(mod._collected_tools)

        # Execute search through call_parent_tool
        result = call_parent_tool_func(name="search", query="test query")
        assert result == "Results for: test query"

    def test_call_parent_tool_not_found(self):
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput

        mod = Parent()
        call_parent_tool_func = mod._generate_parent_tool(mod._collected_tools)

        with pytest.raises(BranchError, match="not found"):
            call_parent_tool_func(name="nonexistent")

    def test_call_parent_tool_available_in_branch(self):
        """Test that call_parent_tool is auto-added to branch during execution."""
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        class Parent(Module):
            model = "test-model"
            tools = [search]
            branches = [FactCheckBranch]
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "OK"},
                "call_branch_finish"
            )

            def check_branch_tools(*args, **kwargs):
                # Check that call_parent_tool is in the tools
                tools = kwargs.get("tools", [])
                tool_names = [t["function"]["name"] for t in tools]
                assert "call_parent_tool" in tool_names, f"call_parent_tool not in tools: {tool_names}"
                return MockResponse(tool_calls=[branch_finish])

            mock_completion.side_effect = check_branch_tools

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            mod._execute_branch(FactCheckBranch, "end_result", claim="test")

    def test_call_parent_tool_schema(self):
        @tool
        def search(query: str) -> str:
            """Search for information."""
            return f"Results for: {query}"

        class Parent(Module):
            model = "test-model"
            tools = [search]
            final_output = AnswerOutput

        mod = Parent()
        pt = mod._generate_parent_tool(mod._collected_tools)
        schema = pt._tool_schema
        assert schema["function"]["name"] == "call_parent_tool"
        assert "additionalProperties" in schema["function"]["parameters"]


# =============================================================================
# Test: Manual branching via self.branch()
# =============================================================================

class TestManualBranching:
    def test_manual_branch_returns_result(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
            max_steps = 3

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "Confirmed"},
                "call_branch_finish"
            )
            mock_completion.return_value = MockResponse(tool_calls=[branch_finish])

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            result = mod.branch(FactCheckBranch, claim="The sky is blue")

            assert isinstance(result, VerificationOutput)
            assert result.verified is True

    def test_manual_branch_injects_to_history(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
            max_steps = 3

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "OK"},
                "call_branch_finish"
            )
            mock_completion.return_value = MockResponse(tool_calls=[branch_finish])

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            history_len_before = len(mod.history)
            mod.branch(FactCheckBranch, claim="test")

            # History should have a new entry with branch result
            assert len(mod.history) > history_len_before
            last_msg = mod.history[-1]
            assert "[Branch Result]" in last_msg["content"]

    def test_manual_branch_with_merge_strategy(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "OK"},
                "call_branch_finish"
            )
            # Two calls: one for branch execution, one for summary
            mock_completion.side_effect = [
                MockResponse(tool_calls=[branch_finish]),
                MockResponse(content="Summary of branch execution"),
            ]

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            result = mod.branch(FactCheckBranch, merge="summarize", claim="test")

            assert result.verified is True
            last_msg = mod.history[-1]
            assert "Branch summary:" in last_msg["content"]

    def test_manual_branch_not_in_branches_dict(self):
        """Manual branching doesn't require the class to be in self.branches."""
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "OK"},
                "call_branch_finish"
            )
            mock_completion.return_value = MockResponse(tool_calls=[branch_finish])

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            # FactCheckBranch is NOT in self.branches, but manual branch still works
            result = mod.branch(FactCheckBranch, claim="test")
            assert result.verified is True


# =============================================================================
# Test: Branch errors
# =============================================================================

class TestBranchErrors:
    def test_branch_execution_failure_raises_branch_error(self):
        class FailingBranch(Module):
            """This branch will fail."""
            model = "test-model"
            initial_input = ClaimInput
            final_output = VerificationOutput

        class Parent(Module):
            model = "test-model"
            branches = [FailingBranch]
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            # Make the branch's LLM call fail
            mock_completion.side_effect = Exception("LLM call failed")

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            with pytest.raises(BranchError, match="Branch execution failed"):
                mod._execute_branch(FailingBranch, "end_result", claim="test")


# =============================================================================
# Test: Nested branching
# =============================================================================

class TestNestedBranching:
    def test_nested_branch_has_own_branch_tool(self):
        """If a branch module has its own branches, it gets a branch tool."""
        class InnerBranch(Module):
            """Inner branch."""
            model = "test-model"
            initial_input = ClaimInput
            final_output = VerificationOutput

        class OuterBranch(Module):
            """Outer branch with nested branches."""
            model = "test-model"
            initial_input = ClaimInput
            final_output = VerificationOutput
            branches = [InnerBranch]

        outer = OuterBranch()
        tool_names = [t.__name__ for t in outer._collected_tools]
        assert "branch" in tool_names

    def test_nested_branch_parent_tool_accesses_immediate_parent(self):
        """Nested branch's call_parent_tool accesses the immediate parent only."""
        @tool
        def outer_search(query: str) -> str:
            """Outer search tool."""
            return f"Outer: {query}"

        @tool
        def inner_search(query: str) -> str:
            """Inner search tool."""
            return f"Inner: {query}"

        class InnerBranch(Module):
            """Inner branch."""
            model = "test-model"
            initial_input = ClaimInput
            final_output = VerificationOutput

        class OuterBranch(Module):
            """Outer branch with nested branches."""
            model = "test-model"
            tools = [inner_search]
            initial_input = ClaimInput
            final_output = VerificationOutput
            branches = [InnerBranch]

        class Parent(Module):
            model = "test-model"
            tools = [outer_search]
            branches = [OuterBranch]
            final_output = AnswerOutput

        # Build call_parent_tool for OuterBranch from Parent's tools
        parent = Parent()
        outer_parent_tool = parent._generate_parent_tool(parent._collected_tools)
        result = json.loads(outer_parent_tool())
        names = [t["name"] for t in result]
        assert "outer_search" in names
        assert "inner_search" not in names


# =============================================================================
# Test: Branch with multi-turn agentic loop
# =============================================================================

class TestBranchMultiTurn:
    def test_multi_turn_branch_execution(self):
        """Test branch that runs a multi-step agentic loop."""
        @tool
        def verify(claim: str) -> str:
            """Verify a claim."""
            return f"Verification: {claim} is true"

        class MultiStepBranch(Module):
            """Multi-step verification branch."""
            model = "test-model"
            initial_input = ClaimInput
            final_output = VerificationOutput
            max_steps = 3
            tools = [verify]

        class Parent(Module):
            model = "test-model"
            branches = [MultiStepBranch]
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            # Branch step 1: call verify
            verify_call = create_tool_call("verify", {"claim": "test"}, "call_v")
            # Branch step 2: call __finish__
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "Verified via tool"},
                "call_bf"
            )

            mock_completion.side_effect = [
                MockResponse(tool_calls=[verify_call]),
                MockResponse(tool_calls=[branch_finish]),
            ]

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            result, content = mod._execute_branch(MultiStepBranch, "end_result", claim="test claim")

            assert result.verified is True
            assert result.explanation == "Verified via tool"


# =============================================================================
# Test: Branch result returned as tool result in agentic loop
# =============================================================================

class TestBranchInAgenticLoop:
    def test_branch_result_returned_as_tool_result(self):
        """Test that branch results flow back properly in the parent loop."""
        class Parent(Module):
            model = "test-model"
            branches = [FactCheckBranch]
            final_output = AnswerOutput
            max_steps = 5

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            # Parent step 1: calls branch
            branch_call = create_tool_call(
                "branch",
                {"name": "FactCheckBranch", "claim": "Earth is round"},
                "call_1"
            )
            # Branch: calls __finish__
            branch_finish = create_tool_call(
                "__finish__",
                {"verified": "true", "explanation": "Confirmed round"},
                "call_bf"
            )
            # Parent step 2: calls __finish__
            parent_finish = create_tool_call(
                "__finish__",
                {"answer": "Earth is confirmed round"},
                "call_pf"
            )

            mock_completion.side_effect = [
                MockResponse(tool_calls=[branch_call]),
                MockResponse(tool_calls=[branch_finish]),
                MockResponse(tool_calls=[parent_finish]),
            ]

            mod = Parent()
            result = mod()

            assert result.answer == "Earth is confirmed round"

            # Check that branch tool result was added to parent history
            tool_msgs = [m for m in mod.history if m.get("role") == "tool"]
            assert len(tool_msgs) > 0
            # At least one tool result should contain the branch output
            branch_results = [m for m in tool_msgs if "verified" in m.get("content", "")]
            assert len(branch_results) > 0


# =============================================================================
# Test: format_branch_history
# =============================================================================

class TestFormatBranchHistory:
    def test_format_skips_system_messages(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
        mod = Parent()

        history = [
            {"role": "system", "content": "should be skipped"},
            {"role": "user", "content": "Hello"},
        ]
        text = mod._format_branch_history(history)
        assert "should be skipped" not in text
        assert "[User] Hello" in text

    def test_format_tool_calls(self):
        class Parent(Module):
            model = "test-model"
            final_output = AnswerOutput
        mod = Parent()

        history = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"function": {"name": "search", "arguments": '{"q": "test"}'}}
            ]},
            {"role": "tool", "content": "Found results"},
        ]
        text = mod._format_branch_history(history)
        assert "Called search" in text
        assert "[Tool Result] Found results" in text


# =============================================================================
# Test: Branch with no initial_input
# =============================================================================

class TestBranchNoInitialInput:
    def test_branch_without_initial_input(self):
        class Parent(Module):
            model = "test-model"
            branches = [NoBranchInput]
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            branch_finish = create_tool_call(
                "__finish__",
                {"answer": "Simple answer"},
                "call_bf"
            )
            mock_completion.return_value = MockResponse(tool_calls=[branch_finish])

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent"},
                {"role": "user", "content": "test"}
            ]
            result, _ = mod._execute_branch(NoBranchInput, "end_result")
            assert result.answer == "Simple answer"


# =============================================================================
# Test: Auto-fallback to summarize when branch has no final_output
# =============================================================================

class TestBranchNoFinalOutputFallback:
    def test_end_result_falls_back_to_summarize_when_no_final_output(self):
        """When a branch has no final_output and merge='end_result',
        it should auto-fallback to 'summarize' instead of returning None."""

        class NoOutputBranch(Module):
            """A branch with no final_output."""
            model = "test-model"
            max_steps = 3
            # No final_output defined

        class Parent(Module):
            model = "test-model"
            branches = [NoOutputBranch]
            final_output = AnswerOutput

        with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
            # Branch calls __finish__ with no args (no final_output → returns None)
            finish_call = create_tool_call("__finish__", {}, "call_fin")
            mock_completion.return_value = MockResponse(tool_calls=[finish_call])

            mod = Parent()
            mod.history = [
                {"role": "system", "content": "Parent system prompt"},
                {"role": "user", "content": "test input"}
            ]

            # Patch _merge_summarize to verify it gets called
            with patch.object(mod, '_merge_summarize', return_value="Branch summary:\nDid research.\n\nFinal result:\nNone") as mock_summarize:
                result, merged = mod._execute_branch(NoOutputBranch, "end_result")

                # result should be None (branch has no final_output)
                assert result is None
                # _merge_summarize should have been called instead of _merge_end_result
                mock_summarize.assert_called_once()
                assert "summary" in merged.lower()
