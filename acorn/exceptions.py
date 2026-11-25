"""Exception classes for Acorn."""


class AcornError(Exception):
    """Base exception for all Acorn errors."""


class ParseError(AcornError):
    """Raised when output validation or parsing fails.

    This exception is raised when the LLM's output cannot be validated
    against the expected Pydantic model schema.

    Attributes:
        message: Human-readable error description
        raw_output: The raw output that failed validation (optional)
    """

    def __init__(self, message: str, raw_output: any = None):
        super().__init__(message)
        self.raw_output = raw_output


class TruncatedOutputError(AcornError):
    """Raised when the model's output was cut off at the token limit.

    This exception is raised when a response has ``finish_reason == "length"``
    and contains a ``__finish__`` tool call. Truncated arguments can still be
    valid JSON (for example ``{}``), so accepting them would silently return
    an empty or partial result.

    Attributes:
        message: Human-readable error description
        token_limit: Configured ``max_tokens``/``max_completion_tokens``, if any
        raw_output: The raw ``__finish__`` arguments that were cut off (optional)
    """

    def __init__(self, message: str, token_limit: int | None = None, raw_output: any = None):
        super().__init__(message)
        self.token_limit = token_limit
        self.raw_output = raw_output


class BranchError(AcornError):
    """Raised when branch execution fails.

    This exception is raised when a branch module encounters an error
    during execution or fails to produce valid output.
    """


class ServiceError(AcornError):
    """Raised when a service lifecycle operation fails.

    This exception is raised when a service's setup(), teardown(),
    or health() method encounters an error.
    """


class ToolConflictError(AcornError):
    """Raised when tool name conflicts are detected.

    This exception is raised when multiple tools with the same name
    are registered, either from the tools list, decorated methods,
    or branches.
    """
