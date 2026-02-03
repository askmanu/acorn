"""Example demonstrating metadata support in Acorn.

This example shows how to use the metadata attribute to track
user sessions, traces, and custom properties in LiteLLM.
"""

from pydantic import BaseModel
from acorn import Module


# Define input/output schemas
class Question(BaseModel):
    query: str


class Answer(BaseModel):
    response: str


# Example 1: Single-turn module with metadata
class QAAgent(Module):
    """Answer user questions based on provided context."""

    model = "gpt-4"
    initial_input = Question
    final_output = Answer

    # Metadata for tracking in LiteLLM
    metadata = {
        "distinct_id": "user_12345",
        "$ai_session_id": "session_abc_123",
        "$ai_trace_id": "trace_xyz_789",
        "tenant_id": "tenant_789",
        "environment": "production"
    }


# Example 2: Multi-turn agent with metadata
class ResearchAgent(Module):
    """Research agent that can use tools to gather information."""

    model = "gpt-4"
    max_steps = 10
    initial_input = Question
    final_output = Answer

    # Different metadata for research context
    metadata = {
        "user_id": "researcher_456",
        "session_id": "research_session_001",
        "project_id": "proj_123",
        "cost_center": "research_dept"
    }


# Example 3: Dynamic metadata per instance
class CustomAgent(Module):
    """Agent that can have different metadata per instance."""

    model = "gpt-4"
    initial_input = Question
    final_output = Answer

    def __init__(self, user_id: str, session_id: str):
        """Initialize with custom metadata."""
        # Set metadata before calling parent init
        self.metadata = {
            "user_id": user_id,
            "session_id": session_id,
            "agent_type": "custom",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        super().__init__()


def main():
    """Demonstrate metadata usage."""

    print("=" * 60)
    print("Example 1: Single-turn QA Agent with static metadata")
    print("=" * 60)

    # Create agent with static metadata
    qa_agent = QAAgent()
    print(f"Agent metadata: {qa_agent.metadata}")
    print("\nNote: All LLM calls from this agent will include this metadata")
    print("for tracking in your LiteLLM observability platform.\n")

    print("=" * 60)
    print("Example 2: Multi-turn Research Agent with metadata")
    print("=" * 60)

    research_agent = ResearchAgent()
    print(f"Agent metadata: {research_agent.metadata}")
    print("\nNote: Each step in the agentic loop will include this metadata,")
    print("allowing you to track the entire research session.\n")

    print("=" * 60)
    print("Example 3: Dynamic metadata per instance")
    print("=" * 60)

    # Create multiple instances with different metadata
    agent1 = CustomAgent(user_id="user_001", session_id="session_001")
    agent2 = CustomAgent(user_id="user_002", session_id="session_002")

    print(f"Agent 1 metadata: {agent1.metadata}")
    print(f"Agent 2 metadata: {agent2.metadata}")
    print("\nNote: Each agent instance can have different metadata,")
    print("allowing per-user or per-session tracking.\n")

    print("=" * 60)
    print("Example 4: Agent without metadata")
    print("=" * 60)

    class SimpleAgent(Module):
        """Simple agent without metadata."""
        model = "gpt-4"
        initial_input = Question
        final_output = Answer
        # No metadata attribute - works as before

    simple_agent = SimpleAgent()
    print(f"Agent metadata: {simple_agent.metadata}")
    print("\nNote: Agents without metadata work exactly as before.")
    print("metadata=None means no metadata is sent to LiteLLM.\n")

    print("=" * 60)
    print("Integration with LiteLLM")
    print("=" * 60)
    print("""
Metadata is automatically forwarded to LiteLLM's completion() call.
This enables tracking in observability platforms like:

- LiteLLM Proxy Server
- Langfuse
- Helicone
- Custom logging solutions

Common metadata fields:
- distinct_id: User identifier
- $ai_session_id: Session identifier for grouping related calls
- $ai_trace_id: Trace identifier for distributed tracing
- Custom fields: Any additional properties for your use case

For more information, see:
https://docs.litellm.ai/docs/completion/metadata
    """)


if __name__ == "__main__":
    main()
