"""Example: Simple Q&A

This example demonstrates the simplest possible Acorn module: a single-turn
question-answering agent that takes a question and returns a structured answer.

No tools, no agentic loop - just structured input and output.
"""

from pydantic import BaseModel, Field
from acorn import Module


# Input schema
class Question(BaseModel):
    text: str = Field(description="The question to answer")


# Output schema
class Answer(BaseModel):
    response: str = Field(description="The answer to the question")
    confidence: str = Field(description="Confidence level: low, medium, or high")


SYSTEM_PROMPT = """You are a helpful assistant that answers questions clearly and concisely.

Provide accurate, well-reasoned answers and indicate your confidence level.
Keep your responses focused and easy to understand."""


class SimpleQA(Module):
    """Simple question-answering assistant."""

    system_prompt = SYSTEM_PROMPT
    initial_input = Question
    final_output = Answer

    # Configuration
    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.7
