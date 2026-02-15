"""Code Copilot Example with Branch for File Changes.

This example demonstrates:
1. A main copilot module with search and read tools
2. A branch module that applies file changes
3. How branches use call_parent_tool() to access parent's tools

The branch can use call_parent_tool() to access the parent's search_files
and read_file tools when needed, while also having its own update_file tool.
"""

from pydantic import BaseModel, Field
from acorn import Module, tool


# =============================================================================
# Simulated file system (for demo purposes)
# =============================================================================

FILE_SYSTEM = {
    "src/main.py": """def greet(name):
    print(f"Hello {name}")

def main():
    greet("World")
""",
    "src/utils.py": """def add(a, b):
    return a + b

def subtract(a, b):
    return a - b
""",
    "tests/test_main.py": """import unittest
from src.main import greet

class TestGreet(unittest.TestCase):
    def test_greet(self):
        greet("Test")
""",
}


# =============================================================================
# Tools for the main copilot
# =============================================================================

@tool
def search_files(pattern: str) -> str:
    """Search for files matching a pattern.

    Args:
        pattern: File pattern to search for (e.g., "*.py", "test_*")

    Returns:
        List of matching file paths
    """
    import fnmatch

    matches = []
    for filepath in FILE_SYSTEM.keys():
        if fnmatch.fnmatch(filepath, f"*{pattern}*"):
            matches.append(filepath)

    if not matches:
        return f"No files found matching pattern: {pattern}"

    return "\n".join(matches)


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a file.

    Args:
        filepath: Path to the file to read

    Returns:
        File contents
    """
    if filepath not in FILE_SYSTEM:
        return f"Error: File '{filepath}' not found"

    return FILE_SYSTEM[filepath]


@tool
def list_all_files() -> str:
    """List all files in the project.

    Returns:
        List of all file paths
    """
    return "\n".join(sorted(FILE_SYSTEM.keys()))


# =============================================================================
# Tools for the branch (file modification)
# =============================================================================

@tool
def update_file(filepath: str, new_content: str) -> str:
    """Update a file with new content.

    Args:
        filepath: Path to the file to update
        new_content: New content for the file

    Returns:
        Success message
    """
    if filepath not in FILE_SYSTEM:
        return f"Error: File '{filepath}' not found"

    FILE_SYSTEM[filepath] = new_content
    return f"Successfully updated {filepath}"


# =============================================================================
# Schema definitions
# =============================================================================

class UserRequest(BaseModel):
    """User's code change request."""
    request: str = Field(description="The code change requested by the user")


class FileChange(BaseModel):
    """Details of a file change to apply."""
    filepath: str = Field(description="Path to the file to modify")
    new_content: str = Field(description="The new content for the file")
    reason: str = Field(description="Why this change is being made")


class ChangeResult(BaseModel):
    """Result of applying a file change."""
    success: bool = Field(description="Whether the change was successful")
    filepath: str = Field(description="The file that was changed")
    message: str = Field(description="Details about what was done")


class CopilotResponse(BaseModel):
    """Final response from the copilot."""
    analysis: str = Field(description="Analysis of the request and changes made")
    files_changed: list[str] = Field(description="List of files that were modified")
    summary: str = Field(description="Summary of all changes")


# =============================================================================
# Branch Module: Applies file changes
# =============================================================================

class ApplyChangeBranch(Module):
    """Apply a file change to the codebase.

    This branch is responsible for safely applying changes to files.
    It can use call_parent_tool() to:
    - Search for related files that might need updates
    - Read current file contents before modifying
    - Access any other parent tools as needed
    """

    initial_input = FileChange
    final_output = ChangeResult
    max_steps = 5

    tools = [update_file]  # Branch-specific tool

    # Note: The branch automatically gets access to parent tools via call_parent_tool():
    # - search_files
    # - read_file
    # - list_all_files


# =============================================================================
# Main Copilot Module
# =============================================================================

class CodeCopilot(Module):
    """AI code copilot that helps with code modifications.

    You are an AI coding assistant. When the user requests changes:
    1. Analyze what needs to be changed
    2. Search for relevant files if needed
    3. Read current file contents
    4. Use the ApplyChangeBranch to make changes safely

    The ApplyChangeBranch can call your tools using call_parent_tool(name="tool_name", ...).
    For example, it might read files before modifying them, or search for related files.
    """

    initial_input = UserRequest
    final_output = CopilotResponse
    max_steps = 10

    tools = [search_files, read_file, list_all_files]
    branches = [ApplyChangeBranch]


# =============================================================================
# Demo
# =============================================================================

def main():
    """Run the code copilot demo."""
    print("=" * 70)
    print("CODE COPILOT WITH BRANCH DEMO")
    print("=" * 70)
    print()

    # Create copilot instance
    copilot = CodeCopilot()

    # Example 1: Simple change request
    print("Example 1: Add a goodbye function to main.py")
    print("-" * 70)

    request = """Add a new function called 'goodbye' to src/main.py that takes a name
parameter and prints 'Goodbye {name}'. Also update main() to call both greet and goodbye."""

    print(f"User Request: {request}")
    print()
    print("Processing...")
    print()

    # Note: In a real implementation, the copilot would:
    # 1. Use read_file to see current contents
    # 2. Call the ApplyChangeBranch with the change
    # 3. The branch might use call_parent_tool(name="read_file", filepath="src/main.py")
    #    to verify the file before updating
    # 4. The branch uses update_file to apply changes
    # 5. Return results

    print("How the branch uses call_parent_tool():")
    print("  - Branch can call: call_parent_tool(name='read_file', filepath='src/main.py')")
    print("  - Branch can call: call_parent_tool(name='search_files', pattern='*.py')")
    print("  - Branch uses its own: update_file(filepath='...', new_content='...')")
    print()

    # Example 2: Show branch accessing parent tools
    print("=" * 70)
    print("Example 2: How branches access parent tools")
    print("-" * 70)
    print()

    # Instantiate the branch manually to demonstrate tool access
    branch = ApplyChangeBranch()

    # In a real scenario, when executing within the parent context,
    # the branch would have call_parent_tool automatically added
    print("Branch tools (from tools attribute):")
    for tool_func in branch.tools:
        print(f"  - {tool_func.__name__}")
    print()

    print("Parent tools (accessible via call_parent_tool):")
    print("  - search_files")
    print("  - read_file")
    print("  - list_all_files")
    print()

    print("The branch calls parent tools like this:")
    print("  call_parent_tool(name='read_file', filepath='src/main.py')")
    print()

    # Example 3: Demonstrate file system state
    print("=" * 70)
    print("Example 3: Current file system state")
    print("-" * 70)
    print()

    print(list_all_files())
    print()

    print("Contents of src/main.py:")
    print("-" * 70)
    print(read_file("src/main.py"))
    print()

    # Show what a file update would look like
    print("=" * 70)
    print("Example 4: Simulated file update")
    print("-" * 70)
    print()

    new_content = """def greet(name):
    print(f"Hello {name}")

def goodbye(name):
    print(f"Goodbye {name}")

def main():
    greet("World")
    goodbye("World")
"""

    result = update_file("src/main.py", new_content)
    print(result)
    print()

    print("Updated contents of src/main.py:")
    print("-" * 70)
    print(read_file("src/main.py"))
    print()

    print("=" * 70)
    print("KEY TAKEAWAYS")
    print("=" * 70)
    print()
    print("1. Main copilot has tools: search_files, read_file, list_all_files")
    print("2. Branch has its own tool: update_file")
    print("3. Branch can access parent tools via call_parent_tool():")
    print("   - call_parent_tool(name='read_file', filepath='...')")
    print("   - call_parent_tool(name='search_files', pattern='...')")
    print("4. This allows branches to extend functionality while still")
    print("   accessing the parent's capabilities")
    print()


if __name__ == "__main__":
    main()
