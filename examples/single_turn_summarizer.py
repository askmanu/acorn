"""Example: Simple Single-Turn Summarizer

This example demonstrates a basic single-turn module that summarizes text
without using an agentic loop.
"""

from pydantic import BaseModel, Field
from acorn import Module


class TextSummarizer(Module):
    """Summarize long text into concise summaries."""

    # Input schema
    class Input(BaseModel):
        text: str = Field(description="The text to summarize")
        max_words: int = Field(
            default=100,
            description="Maximum words in summary"
        )

    # Output schema
    class Output(BaseModel):
        summary: str = Field(description="The concise summary")
        word_count: int = Field(description="Number of words in summary")
        main_topics: list[str] = Field(description="Main topics identified")

    initial_input = Input
    final_output = Output

    # Configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.5
    max_tokens = 2000
    # max_steps = None means single-turn mode (default)


# Usage example (would require actual API key):
if __name__ == "__main__":
    # This would require ANTHROPIC_API_KEY to be set
    summarizer = TextSummarizer()

    sample_text = """
    Large language models (LLMs) have revolutionized natural language processing
    in recent years. These models, trained on vast amounts of text data, can
    perform a wide variety of tasks including translation, summarization, question
    answering, and even code generation. The most prominent examples include
    GPT-4, Claude, and PaLM. However, they also raise important questions about
    AI safety, bias, and environmental impact due to their computational requirements.
    """

    # Example call:
    # result = summarizer(text=sample_text, max_words=50)
    #
    # print("Summary:", result.summary)
    # print(f"Word count: {result.word_count}")
    # print("Topics:", ", ".join(result.main_topics))

    print("Example summarizer created successfully!")
    print(f"Mode: {'Single-turn' if summarizer.max_steps is None else 'Multi-turn'}")
