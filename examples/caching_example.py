"""Example demonstrating provider caching configuration.

Prompt caching can significantly reduce costs and latency for requests
with repeated content (like system prompts and initial context).

See LiteLLM documentation: https://docs.litellm.ai/docs/tutorials/prompt_caching
"""

from pydantic import BaseModel
from acorn import Module


class Question(BaseModel):
    text: str


class Answer(BaseModel):
    response: str


# Example 1: Default caching (recommended for most use cases)
class AnalystWithDefaultCache(Module):
    """You are a helpful data analyst.

    You provide clear, concise answers to questions about data analysis.
    Always cite your reasoning and provide examples when relevant.
    """
    initial_input = Question
    final_output = Answer
    cache = True  # Caches system message and first user message


# Example 2: Custom caching (advanced usage)
class AnalystWithCustomCache(Module):
    """You are a helpful data analyst.

    You provide clear, concise answers to questions about data analysis.
    Always cite your reasoning and provide examples when relevant.
    """
    initial_input = Question
    final_output = Answer
    # Only cache the system message
    cache = [{"location": "message", "role": "system"}]


# Example 3: No caching (default behavior)
class AnalystWithoutCache(Module):
    """You are a helpful data analyst.

    You provide clear, concise answers to questions about data analysis.
    Always cite your reasoning and provide examples when relevant.
    """
    initial_input = Question
    final_output = Answer
    # cache = None (default - no caching)


# Usage
if __name__ == "__main__":
    # Default caching
    print("=== Default Caching ===")
    print("Caches both system message and first user message")
    print("Most cost-effective for repeated queries with similar context\n")

    # Custom caching
    print("=== Custom Caching ===")
    print("Caches only system message")
    print("Useful when user messages vary significantly\n")

    # No caching
    print("=== No Caching ===")
    print("Standard behavior - no caching")
    print("Use when messages are always unique\n")

    print("Note: Actual caching behavior depends on provider support.")
    print("LiteLLM automatically handles provider-specific caching APIs.")

    # Example: Multiple queries with same system prompt benefit from caching
    analyst = AnalystWithDefaultCache()

    # First query - no cache hit
    # result1 = analyst(text="What is the median?")

    # Second query - cache hit on system prompt + first message pattern
    # result2 = analyst(text="What is standard deviation?")
