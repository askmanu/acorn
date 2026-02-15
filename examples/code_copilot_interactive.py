"""Interactive Code Copilot Example with Branch Execution.

This example shows a complete working implementation where:
1. The copilot analyzes code change requests
2. Uses a branch to safely apply file changes
3. The branch uses call_parent_tool() to access parent's read/search capabilities

Run with: python examples/code_copilot_interactive.py
"""

from pydantic import BaseModel, Field
from acorn import Module, tool
from unittest.mock import patch, MagicMock
import json


# =============================================================================
# Simulated file system
# =============================================================================

FILE_SYSTEM = {
    "app.py": """def calculate(x, y):
    return x + y

def main():
    result = calculate(5, 3)
    print(result)
""",
    "utils.py": """def format_output(value):
    return f"Result: {value}"
""",
}


# =============================================================================
# Parent Tools
# =============================================================================

@tool
def search_files(pattern: str) -> str:
    """Search for files matching a pattern."""
    import fnmatch
    matches = [f for f in FILE_SYSTEM.keys() if fnmatch.fnmatch(f, f"*{pattern}*")]
    return "\n".join(matches) if matches else "No files found"


@tool
def read_file(filepath: str) -> str:
    """Read the contents of a file."""
    if filepath not in FILE_SYSTEM:
        return f"Error: File '{filepath}' not found"
    return FILE_SYSTEM[filepath]


# =============================================================================
# Branch Tools
# =============================================================================

@tool
def write_file(filepath: str, content: str) -> str:
    """Write content to a file."""
    FILE_SYSTEM[filepath] = content
    return f"Successfully wrote to {filepath}"


# =============================================================================
# Schemas
# =============================================================================

class ChangeRequest(BaseModel):
    """File change to apply."""
    filepath: str = Field(description="Path to the file to modify")
    new_content: str = Field(description="New content for the file")


class ChangeResult(BaseModel):
    """Result of applying changes."""
    success: bool
    filepath: str
    message: str


class UserInput(BaseModel):
    """User input to the copilot."""
    request: str = Field(default="", description="The user's request")


class CopilotResult(BaseModel):
    """Final copilot output."""
    summary: str = Field(description="Summary of what was done")
    files_modified: list[str] = Field(description="List of modified files")


# =============================================================================
# Branch: Applies changes safely
# =============================================================================

class FileChangeBranch(Module):
    """Apply file changes safely.

    You are a file modification assistant. When asked to modify a file:
    1. First, use call_parent_tool to read the current file contents
    2. Verify the change makes sense
    3. Apply the change using write_file
    4. Confirm success

    You can access parent tools:
    - call_parent_tool(name="read_file", filepath="...")
    - call_parent_tool(name="search_files", pattern="...")
    """

    initial_input = ChangeRequest
    final_output = ChangeResult
    max_steps = 5
    tools = [write_file]


# =============================================================================
# Main Copilot
# =============================================================================

class CodeCopilot(Module):
    """AI code copilot for making code changes.

    You help users modify their code. When a user requests a change:
    1. Understand what needs to be changed
    2. Search for the right file if needed
    3. Read the current contents
    4. Use the FileChangeBranch to apply the change
    5. Return a summary of what was done

    Use the 'branch' tool to delegate file modifications to the FileChangeBranch.
    """

    initial_input = UserInput
    final_output = CopilotResult
    max_steps = 10
    tools = [search_files, read_file]
    branches = [FileChangeBranch]


# =============================================================================
# Mock LLM Responses
# =============================================================================

def create_mock_tool_call(name: str, arguments: dict, call_id: str = "call_123"):
    """Helper to create mock tool calls."""
    tc = MagicMock()
    tc.id = call_id
    tc.type = "function"
    tc.function = MagicMock()
    tc.function.name = name
    tc.function.arguments = json.dumps(arguments)
    return tc


def create_mock_response(content=None, tool_calls=None):
    """Helper to create mock LLM responses."""
    response = MagicMock()
    response.choices = [MagicMock()]
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls or []
    response.choices[0].message = message
    response.choices[0].finish_reason = "stop"
    return response


# =============================================================================
# Demo Scenarios
# =============================================================================

def demo_with_mocked_llm():
    """Demo with mocked LLM responses to show the workflow."""
    print("=" * 70)
    print("CODE COPILOT DEMO - Mocked LLM Workflow")
    print("=" * 70)
    print()

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Scenario: User asks to add a multiply function to app.py

        # Step 1: Copilot reads the file
        step1_call = create_mock_tool_call(
            "read_file",
            {"filepath": "app.py"},
            "call_1"
        )

        # Step 2: Copilot calls branch to apply changes
        step2_call = create_mock_tool_call(
            "branch",
            {
                "name": "FileChangeBranch",
                "filepath": "app.py",
                "new_content": """def calculate(x, y):
    return x + y

def multiply(x, y):
    return x * y

def main():
    result = calculate(5, 3)
    print(result)
    result2 = multiply(4, 2)
    print(result2)
"""
            },
            "call_2"
        )

        # Step 3: Branch reads file using call_parent_tool (in branch context)
        branch_step1_call = create_mock_tool_call(
            "call_parent_tool",
            {"name": "read_file", "filepath": "app.py"},
            "call_branch_1"
        )

        # Step 4: Branch writes the file
        branch_step2_call = create_mock_tool_call(
            "write_file",
            {
                "filepath": "app.py",
                "content": """def calculate(x, y):
    return x + y

def multiply(x, y):
    return x * y

def main():
    result = calculate(5, 3)
    print(result)
    result2 = multiply(4, 2)
    print(result2)
"""
            },
            "call_branch_2"
        )

        # Step 5: Branch finishes
        branch_finish_call = create_mock_tool_call(
            "__finish__",
            {
                "success": True,
                "filepath": "app.py",
                "message": "Added multiply function to app.py"
            },
            "call_branch_finish"
        )

        # Step 6: Copilot finishes
        copilot_finish_call = create_mock_tool_call(
            "__finish__",
            {
                "summary": "Added a multiply function to app.py and updated main() to use it",
                "files_modified": ["app.py"]
            },
            "call_finish"
        )

        # Set up mock responses in sequence
        mock_completion.side_effect = [
            create_mock_response(tool_calls=[step1_call]),  # Copilot reads file
            create_mock_response(tool_calls=[step2_call]),  # Copilot calls branch
            create_mock_response(tool_calls=[branch_step1_call]),  # Branch reads via parent
            create_mock_response(tool_calls=[branch_step2_call]),  # Branch writes
            create_mock_response(tool_calls=[branch_finish_call]),  # Branch finishes
            create_mock_response(tool_calls=[copilot_finish_call]),  # Copilot finishes
        ]

        # Execute
        copilot = CodeCopilot()

        print("User Request: Add a multiply function to app.py")
        print()
        print("Executing copilot...")
        print()

        try:
            result = copilot(request="Add a multiply function to app.py")

            print("✓ Execution completed successfully!")
            print()
            print(f"Summary: {result.summary}")
            print(f"Files Modified: {', '.join(result.files_modified)}")
            print()

        except Exception as e:
            print(f"Note: Full execution would require actual LLM calls")
            print(f"Error: {e}")
            print()

        # Show the call sequence
        print("Call Sequence:")
        print("-" * 70)
        print("1. Copilot → read_file(filepath='app.py')")
        print("2. Copilot → branch(name='FileChangeBranch', ...)")
        print("3.   Branch → call_parent_tool(name='read_file', filepath='app.py')")
        print("4.   Branch → write_file(filepath='app.py', content='...')")
        print("5.   Branch → __finish__(success=True, ...)")
        print("6. Copilot → __finish__(summary='...', files_modified=['app.py'])")
        print()


def demo_manual_branch_call():
    """Demo showing manual branch execution."""
    print("=" * 70)
    print("MANUAL BRANCH EXECUTION DEMO")
    print("=" * 70)
    print()

    with patch('acorn.llm.litellm_client.litellm.completion') as mock_completion:
        # Mock the branch's internal steps
        branch_read_call = create_mock_tool_call(
            "call_parent_tool",
            {"name": "read_file", "filepath": "app.py"},
            "call_1"
        )

        branch_write_call = create_mock_tool_call(
            "write_file",
            {"filepath": "app.py", "content": "# Modified content"},
            "call_2"
        )

        branch_finish_call = create_mock_tool_call(
            "__finish__",
            {
                "success": True,
                "filepath": "app.py",
                "message": "File updated successfully"
            },
            "call_3"
        )

        mock_completion.side_effect = [
            create_mock_response(tool_calls=[branch_read_call]),
            create_mock_response(tool_calls=[branch_write_call]),
            create_mock_response(tool_calls=[branch_finish_call]),
        ]

        # Create copilot and manually trigger branch
        copilot = CodeCopilot()
        copilot.history = [
            {"role": "system", "content": "You are a code copilot"},
            {"role": "user", "content": "Modify app.py"}
        ]

        print("Manually executing FileChangeBranch from copilot...")
        print()

        try:
            result = copilot.branch(
                FileChangeBranch,
                filepath="app.py",
                new_content="# Modified content"
            )

            print(f"✓ Branch execution completed!")
            print(f"  Success: {result.success}")
            print(f"  File: {result.filepath}")
            print(f"  Message: {result.message}")
            print()

        except Exception as e:
            print(f"Note: This demo shows the structure")
            print(f"Actual execution requires proper LLM setup")
            print()

        print("Key Point:")
        print("-" * 70)
        print("The branch automatically has access to call_parent_tool(),")
        print("which allows it to call the parent's read_file and search_files tools.")
        print()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print()
    demo_with_mocked_llm()
    print()
    demo_manual_branch_call()
    print()
    print("=" * 70)
    print("KEY CONCEPTS DEMONSTRATED")
    print("=" * 70)
    print()
    print("1. Parent copilot has: search_files, read_file tools")
    print("2. Branch has: write_file tool")
    print("3. Branch uses call_parent_tool() to access parent tools:")
    print("   → call_parent_tool(name='read_file', filepath='...')")
    print("   → call_parent_tool(name='search_files', pattern='...')")
    print("4. This separation allows:")
    print("   - Parent focuses on analysis and coordination")
    print("   - Branch focuses on safe file modification")
    print("   - Branch can still access parent's search/read capabilities")
    print()
