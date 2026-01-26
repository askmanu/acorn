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


class BranchError(AcornError):
    """Raised when branch execution fails.

    This exception is raised when a branch module encounters an error
    during execution or fails to produce valid output.
    """


class ToolConflictError(AcornError):
    """Raised when tool name conflicts are detected.

    This exception is raised when multiple tools with the same name
    are registered, either from the tools list, decorated methods,
    or branches.
    """
