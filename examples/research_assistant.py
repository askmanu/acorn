"""Example: Research Assistant with Agentic Loop

This example demonstrates a research assistant that can search for information,
analyze it, and produce a structured report using the agentic loop.
"""

from pydantic import BaseModel, Field
from acorn import Module, tool


class ResearchAssistant(Module):
    """An AI research assistant that gathers and analyzes information."""

    # Input schema
    class Input(BaseModel):
        topic: str = Field(description="The research topic to investigate")
        depth: str = Field(description="Research depth: 'shallow' or 'deep'")

    # Output schema
    class Output(BaseModel):
        summary: str = Field(description="Summary of findings")
        key_points: list[str] = Field(description="Key points discovered")
        sources: list[str] = Field(description="Sources consulted")
        confidence: str = Field(description="Confidence level: low, medium, or high")

    initial_input = Input
    final_output = Output

    # Configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.3
    max_steps = 5

    # Tools available to the agent
    @tool
    def search_web(self, query: str) -> dict:
        """Search the web for information.

        Args:
            query: The search query

        Returns:
            Dictionary with search results
        """
        # In a real implementation, this would call a search API
        return {
            "query": query,
            "results": [
                f"Result 1 for '{query}'",
                f"Result 2 for '{query}'",
                f"Result 3 for '{query}'"
            ]
        }

    @tool
    def search_papers(self, keywords: str) -> list:
        """Search academic papers.

        Args:
            keywords: Keywords to search for

        Returns:
            List of paper titles
        """
        # In a real implementation, this would call an academic API
        return [
            f"Paper about {keywords} - Study 1",
            f"Paper about {keywords} - Study 2"
        ]

    @tool
    def analyze_data(self, data: str) -> str:
        """Analyze collected data.

        Args:
            data: The data to analyze

        Returns:
            Analysis results
        """
        # Simplified analysis
        return f"Analysis of: {data[:100]}..."

    def on_step(self, step):
        """Called after each step in the loop."""
        print(f"\n--- Step {step.counter} ---")
        print(f"Tools called: {[tc.name for tc in step.tool_calls]}")

        for result in step.tool_results:
            if result.error:
                print(f"Error in {result.name}: {result.error}")
            else:
                print(f"{result.name} returned: {str(result.output)[:100]}...")

        return step


# Usage example (would require actual API key):
if __name__ == "__main__":
    # This would require ANTHROPIC_API_KEY to be set
    assistant = ResearchAssistant()

    # Example call:
    # result = assistant(
    #     topic="Large Language Models",
    #     depth="shallow"
    # )
    #
    # print("\n=== Research Report ===")
    # print(f"Summary: {result.summary}")
    # print(f"\nKey Points:")
    # for point in result.key_points:
    #     print(f"  - {point}")
    # print(f"\nSources: {', '.join(result.sources)}")
    # print(f"Confidence: {result.confidence}")

    print("Example module created successfully!")
    print(f"Tools available: {[t.__name__ for t in assistant._collected_tools]}")
