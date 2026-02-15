# Code Copilot Examples

These examples demonstrate the branching feature with `call_parent_tool()` in a code copilot scenario.

## Examples

### 1. `code_copilot_with_branch.py` - Conceptual Demo

A simplified example showing the architecture:
- **Parent Module**: Code copilot with search and read capabilities
- **Branch Module**: File modification specialist with write capability
- **Key Feature**: Branch uses `call_parent_tool()` to access parent's tools

**Run:**
```bash
python examples/code_copilot_with_branch.py
```

**What it demonstrates:**
- How branches inherit parent context
- How branches define their own tools
- How `call_parent_tool()` provides access to parent capabilities
- Simulated file system for demonstration

### 2. `code_copilot_interactive.py` - Working Implementation

A more complete example with actual execution flow:
- Shows the full call sequence through the copilot and branch
- Demonstrates both automatic (via `branch` tool) and manual branch invocation
- Uses mocked LLM responses to show the workflow

**Run:**
```bash
python examples/code_copilot_interactive.py
```

**What it demonstrates:**
- Complete execution flow with mocked LLM calls
- How the copilot delegates file changes to the branch
- How the branch uses `call_parent_tool()` during execution
- Manual branch invocation with `copilot.branch()`

## Key Concepts

### Parent Tools vs Branch Tools

**Parent (CodeCopilot) has:**
- `search_files(pattern)` - Find files matching a pattern
- `read_file(filepath)` - Read file contents
- `list_all_files()` - List all files

**Branch (FileChangeBranch / ApplyChangeBranch) has:**
- `write_file(filepath, content)` - Write file contents
- `update_file(filepath, content)` - Update file contents

### How call_parent_tool() Works

When a branch executes, it automatically gets a `call_parent_tool` function that allows it to call any of the parent's tools:

```python
# Inside a branch's execution:
# List available parent tools
call_parent_tool()

# Call a specific parent tool
call_parent_tool(name="read_file", filepath="app.py")
call_parent_tool(name="search_files", pattern="*.py")
```

### Why This Design?

This separation of concerns provides:

1. **Safety**: File writes are isolated in the branch
2. **Reusability**: The branch can still use parent's read/search tools
3. **Clarity**: Each module has a focused purpose
4. **Flexibility**: Branch can access parent tools when needed

## Architecture

```
┌─────────────────────────────────────┐
│        CodeCopilot (Parent)         │
│  - search_files()                   │
│  - read_file()                      │
│  - list_all_files()                 │
└──────────────┬──────────────────────┘
               │
               │ spawns via branch() tool
               │
               ▼
┌─────────────────────────────────────┐
│    FileChangeBranch (Branch)        │
│  - write_file()                     │
│  + call_parent_tool()               │
│    → can call parent's tools        │
└─────────────────────────────────────┘
```

## Workflow Example

1. User: "Add a multiply function to app.py"
2. Copilot uses `read_file("app.py")` to see current contents
3. Copilot calls `branch(name="FileChangeBranch", filepath="app.py", new_content="...")`
4. Branch uses `call_parent_tool(name="read_file", filepath="app.py")` to verify
5. Branch uses `write_file("app.py", new_content)` to apply changes
6. Branch returns success result to parent
7. Copilot returns summary to user

## Extending These Examples

You can extend these examples by:

- Adding more parent tools (e.g., `run_tests`, `lint_code`)
- Adding more branch tools (e.g., `backup_file`, `validate_syntax`)
- Creating multiple branches for different tasks (e.g., `TestGeneratorBranch`, `RefactorBranch`)
- Using real LLM calls instead of mocks

## Notes

- The name `call_parent_tool` (not `parent_tool`) was chosen to avoid conflicts with user-defined tools
- Branches automatically get `call_parent_tool` added during execution
- Branches inherit the parent's conversation history
- Branches use their own system prompts (not the parent's)
