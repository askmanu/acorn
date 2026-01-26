"""Core data types for Acorn."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """Represents a tool call made by the LLM.

    Attributes:
        id: Unique identifier for this tool call
        name: Name of the tool being called
        arguments: Dictionary of arguments passed to the tool
    """
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    """Represents the result of a tool execution.

    Attributes:
        id: Tool call ID this result corresponds to
        name: Name of the tool that was executed
        output: The return value from the tool
        error: Error message if the tool execution failed (None if successful)
    """
    id: str
    name: str
    output: Any
    error: str | None = None


@dataclass
class StreamChunk:
    """Represents a chunk of streamed output.

    Attributes:
        content: Text content being streamed (for normal responses)
        partial: Partial structured output (Partial[T] instance)
        tool_call: Tool call information being streamed
        done: Whether streaming is complete
    """
    content: str | None = None
    partial: Any | None = None  # Will be Partial[T] when streaming __finish__
    tool_call: dict | None = None
    done: bool = False


class Step:
    """Represents a single step in the agentic loop.

    The Step object is mutable and can be modified during on_step callbacks
    to influence the next iteration of the loop.

    Attributes:
        counter: Step number (1-indexed)
        model: Model identifier or config used for this step
        temperature: Temperature parameter for this step
        max_tokens: Maximum tokens for this step
        tools: List of available tools
        response: Raw response from the LLM
        tool_calls: List of ToolCall objects from this step
        tool_results: List of ToolResult objects from tool executions
    """

    def __init__(
        self,
        counter: int,
        model: str | dict,
        temperature: float,
        max_tokens: int,
        tools: list,
        response: dict,
        tool_calls: list[ToolCall],
        tool_results: list[ToolResult],
    ):
        self.counter = counter
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.tools = tools
        self.response = response
        self.tool_calls = tool_calls
        self.tool_results = tool_results

        # Internal flags for step control
        self._finished = False
        self._finish_kwargs = {}
        self._tools_to_add = []
        self._tools_to_remove = []

    def add_tool(self, tool) -> None:
        """Add a tool to be available in the next step.

        Args:
            tool: A function decorated with @tool or a tool function
        """
        self._tools_to_add.append(tool)

    def remove_tool(self, name: str) -> None:
        """Remove a tool by name from the next step.

        Args:
            name: Name of the tool to remove
        """
        self._tools_to_remove.append(name)

    def finish(self, **kwargs) -> None:
        """Terminate the agentic loop and return final output.

        This method signals that the loop should terminate and return
        the provided kwargs as the final output, validated against
        the module's final_output schema.

        Args:
            **kwargs: Fields for the final output model
        """
        self._finished = True
        self._finish_kwargs = kwargs
