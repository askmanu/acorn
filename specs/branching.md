# Branching

Branching allows a module to spawn sub-agents that run independently and return results to the parent. Branches are specialized modules that inherit context from their parent and add focused capabilities for specific tasks.

## When to use branching

Use branching when you need to:

- **Delegate specialized analysis**: Spawn an expert agent for fact-checking, deep research, or verification
- **Parallel processing**: Analyze multiple items concurrently (map-reduce pattern)
- **Focus context**: Create a sub-agent with a narrower scope without polluting the parent's prompt
- **Isolate complex tasks**: Break down multi-step operations into manageable sub-workflows

## Quick example

```python
from acorn import Module
from pydantic import BaseModel, Field

class VerificationOutput(BaseModel):
    verified: bool
    explanation: str

class FactCheckBranch(Module):
    """Verify factual claims using available tools."""
    model = "anthropic/claude-sonnet-4-6"
    final_output = VerificationOutput
    max_steps = 5

class ResearchAgent(Module):
    """Answer questions with fact-checking support."""
    model = "anthropic/claude-haiku-4-5"
    branches = [FactCheckBranch]  # Register the branch
    max_steps = 10
    # Agent can now call branch(name="FactCheckBranch", ...)
```

## Defining a branch module

A branch is a standard `Module` subclass with two requirements:

1. Define `final_output` (required for returning structured results)
2. Optionally define `initial_input` to accept explicit parameters

```python
from acorn import Module, tool
from pydantic import BaseModel, Field

class ClaimInput(BaseModel):
    claim: str = Field(description="The claim to verify")

class VerificationOutput(BaseModel):
    verified: bool = Field(description="Whether the claim is verified")
    evidence: list[str] = Field(description="Supporting evidence")
    confidence: float = Field(description="Confidence score 0-1")

@tool
def search_academic(query: str) -> list[str]:
    """Search academic sources."""
    return ["source1", "source2"]

class FactCheckBranch(Module):
    """Verify factual claims using available tools."""
    
    model = "anthropic/claude-sonnet-4-6"
    temperature = 0.2
    max_steps = 5
    
    initial_input = ClaimInput  # Accept explicit parameters
    final_output = VerificationOutput  # Required - defines return type
    
    tools = [search_academic]  # Branch-specific tools
```

**Key points:**

- `initial_input`: Defines what parameters the branch accepts when called
- `final_output`: Required - must be a Pydantic model
- `tools`: Branch-specific tools that augment parent's tools
- All other Module settings (model, temperature, max_steps, etc.) can be customized

## Registering branches

Register branches in the parent module using the `branches` class attribute:

```python
class ResearchAgent(Module):
    """Main research agent with fact-checking capability."""
    
    model = "anthropic/claude-haiku-4-5"
    max_steps = 15
    
    branches = [FactCheckBranch]  # List of branch classes
```

**Important**: Use a list, not a dict. Each branch class name becomes the branch identifier.

## The branch() tool

When you register branches, Acorn automatically generates a `branch()` tool that the parent agent can call. The tool has two modes:

### Discovery mode (no arguments)

Call `branch()` with no arguments to list available branches and their input schemas:

```python
# Agent calls: branch()
# Returns XML listing:
# <available_branches>
#   <branches>
#     <name>FactCheckBranch</name>
#     <description>Verify factual claims using available tools.</description>
#     <input_schema>
#       <name>claim</name>
#       <required>true</required>
#       <description>The claim to verify</description>
#     </input_schema>
#   </branches>
# </available_branches>
```

### Execution mode (with name + parameters)

Call `branch(name="ClassName", ...)` with the branch name and required parameters:

```python
# Agent calls: branch(name="FactCheckBranch", claim="Python was created in 1991")
# Spawns FactCheckBranch with inherited context
# Returns structured result after branch completes
```

**Parameters:**

- `name` (required): Branch class name to spawn
- `merge` (optional): Merge strategy - `"end_result"` (default) or `"summarize"`
- Additional parameters: Passed to the branch's `initial_input`

## Branch inheritance model

When a branch is spawned, it inherits from its parent:

| Aspect | Inheritance Behavior |
|--------|---------------------|
| **System prompt** | Branch uses its own `system_prompt` (not inherited) |
| **Tools** | Parent tools + branch tools + `call_parent_tool()` + `__finish__` |
| **History** | Full parent conversation history (copied, not referenced) |
| **Context** | Branch sees everything the parent has discussed |
| **Output** | Must call `__finish__()` with `final_output` schema |

```
Parent Module
  │
  ├─ System: "You are a research assistant..."
  ├─ Tools: [search_web, calculate]
  ├─ History: [user: "Is Python from 1991?", ...]
  │
  └─ Spawns branch(name="FactCheckBranch", claim="Python 1991")
      │
      ┌─────────────────────────────────────────────────────┐
      │ FactCheckBranch                                     │
      │                                                     │
      │ System: "Verify factual claims..."                 │
      │ Tools: [search_web, calculate, search_academic,    │
      │         call_parent_tool, __finish__]              │
      │ History: [full parent history] + branch input      │
      │                                                     │
      │ ├─ Step 1: search_academic("Python creation")      │
      │ ├─ Step 2: call_parent_tool(name="search_web", ...) │
      │ └─ Step 3: __finish__(verified=True, ...)          │
      └─────────────────────────────────────────────────────┘
           │
           │ Returns: <branch_result>
           │            <verified>true</verified>
           │            <evidence>...</evidence>
           │          </branch_result>
           ▼
      Parent continues with branch result
```

## Accessing parent tools: call_parent_tool()

Branches automatically receive a `call_parent_tool()` tool that provides access to the parent's tools.

**Discovery mode (no arguments):**

```python
# Branch calls: call_parent_tool()
# Returns JSON listing parent tools:
# [
#   {"name": "search_web", "schema": {...}},
#   {"name": "calculate", "schema": {...}}
# ]
```

**Execution mode (with name + arguments):**

```python
# Branch calls: call_parent_tool(name="search_web", query="Python history")
# Executes parent's search_web tool
# Returns the tool's result
```

**Note**: `call_parent_tool()` filters out `__finish__` and `branch` from the parent's tool list - branches only access regular parent tools.

## Merge strategies

Merge strategies control how branch results are returned to the parent:

### end_result (default)

Returns only the branch's final output as XML:

```python
# Branch finishes with: __finish__(verified=True, explanation="Confirmed")
# Parent receives:
# <branch_result>
#   <verified>true</verified>
#   <explanation>Confirmed</explanation>
# </branch_result>
```

Best for:
- Simple delegation where you only need the final answer
- Fact-checking and verification tasks
- When branch's internal steps don't matter to the parent

### summarize

Uses an LLM to summarize the branch's execution history and final result:

```python
# Branch performs multiple steps and finishes
# Parent receives:
# Branch summary:
# The branch searched academic sources, cross-referenced with Wikipedia,
# and verified the claim was accurate.
#
# Final result:
# {"verified": true, "explanation": "..."}
```

Best for:
- Complex branch workflows where context matters
- Debugging and transparency
- When the parent needs to understand how the conclusion was reached

**Usage:**

```python
# In branch() tool call:
branch(name="FactCheckBranch", merge="summarize", claim="...")
```

**Auto-fallback**: If a branch has no `final_output` defined and `merge="end_result"`, Acorn automatically falls back to `"summarize"` to provide useful context.

## Manual branching: self.branch()

Spawn branches programmatically in callbacks using `self.branch()`:

```python
class ResearchAgent(Module):
    model = "anthropic/claude-haiku-4-5"
    max_steps = 15
    
    def on_step(self, step):
        # Check if verification is needed based on tool results
        for result in step.tool_results:
            if result.name == "search_web" and "unverified" in str(result.output):
                # Manually spawn verification branch
                verification = self.branch(
                    FactCheckBranch,
                    merge="end_result",
                    claim=result.output
                )
                
                # Result is automatically injected into history
                # Continue processing with verification context
        
        return step
```

**Method signature:**

```python
def branch(
    self,
    module_class,  # Branch Module class (positional-only)
    /,
    merge="end_result",  # Merge strategy
    **kwargs  # Arguments for branch's initial_input
) -> BaseModel | None
```

**Returns**: The branch's `final_output` instance (or `None` if no `final_output` defined)

**Behavior**: The branch result is automatically injected into the parent's history as a user message with `[Branch Result]` prefix.

### When to use manual vs declarative

| Scenario | Approach |
|----------|----------|
| Let the agent decide when to branch | Declarative (`branches = [...]`) |
| Branch based on tool results or logic | Manual (`self.branch()` in `on_step`) |
| Conditional branching | Manual |
| Parallel processing (map-reduce) | Declarative |
| Custom history manipulation before branching | Manual |

## Branches without initial_input

Branches can omit `initial_input` to rely solely on inherited history:

```python
class SummaryBranch(Module):
    """Summarize the discussion so far."""
    model = "anthropic/claude-haiku-4-5"
    final_output = SummaryOutput
    # No initial_input - uses inherited history for context
```

This is useful for:
- Summarization tasks
- Meta-analysis of the conversation
- Decision-making based on accumulated context

## Nested branching

Branches can define their own branches:

```python
class DeepAnalysisBranch(Module):
    """Perform deep analysis with fact-checking."""
    model = "anthropic/claude-sonnet-4-6"
    branches = [FactCheckBranch]  # Nested branch
    final_output = AnalysisOutput
```

**Inheritance chain**: Nested branches inherit from their immediate parent branch, which inherited from the root parent. Tools and context accumulate through all levels.

**Caution**: Each nesting level adds latency and cost. Keep nesting depth minimal (1-2 levels recommended).

## Branch errors

If a branch fails during execution:

**Declarative branching** (`branches = [...]`):
- Error returned to parent as tool result
- Agent sees error message and can continue or retry

```
branch() returned error: Branch execution failed: API timeout
```

**Manual branching** (`self.branch()`):
- Exception raised (BranchError)
- Handle in your callback:

```python
def on_step(self, step):
    try:
        result = self.branch(RiskyBranch, data="test")
    except BranchError as e:
        # Log error, add to history, or handle gracefully
        self.history.append({
            "role": "user",
            "content": f"Branch failed: {e}"
        })
    return step
```

## Complete example: Map-reduce pattern

```python
from acorn import Module, tool
from pydantic import BaseModel, Field
import requests

# --- Branch: Analyze one package ---

class PackageInput(BaseModel):
    package_name: str
    ecosystem: str  # "pip" or "npm"

class PackageProfile(BaseModel):
    name: str
    purpose: str
    categories: list[str]

@tool
def fetch_package_info(package_name: str, ecosystem: str) -> dict:
    """Fetch package info from PyPI or npm registry."""
    if ecosystem == "pip":
        url = f"https://pypi.org/pypi/{package_name}/json"
    else:
        url = f"https://registry.npmjs.org/{package_name}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()

class PackageAnalyzerBranch(Module):
    """Analyze a single package."""
    model = "anthropic/claude-sonnet-4-6"
    initial_input = PackageInput
    final_output = PackageProfile
    tools = [fetch_package_info]
    max_steps = 5

# --- Parent: Orchestrate analysis of all packages ---

class DependencyFileInput(BaseModel):
    file_content: str

class PruningPlan(BaseModel):
    redundancy_groups: list[dict]
    packages_to_remove: list[str]

class DependencyScanner(Module):
    """Scan dependencies for redundancies."""
    
    model = "anthropic/claude-haiku-4-5"
    initial_input = DependencyFileInput
    final_output = PruningPlan
    branches = [PackageAnalyzerBranch]
    max_steps = 60
    
    system_prompt = """Parse the dependency file and extract all package names.
    For each package, spawn PackageAnalyzerBranch to analyze it (you can do multiple in parallel).
    After all branches complete, group packages by overlapping purpose and recommend which to remove."""

# --- Usage ---

scanner = DependencyScanner()
result = scanner(file_content=open("requirements.txt").read())
print(f"Packages to remove: {result.packages_to_remove}")
```

**Flow:**
1. Agent parses file → finds 20 packages
2. Agent calls `branch(name="PackageAnalyzerBranch", ...)` 20 times (in parallel)
3. Each branch fetches package info and analyzes purpose
4. Agent receives 20 PackageProfile results
5. Agent groups by overlap and generates PruningPlan

## Common patterns

### Fact-checking

```python
class FactCheckBranch(Module):
    """Verify claims with multiple sources."""
    model = "anthropic/claude-sonnet-4-6"
    initial_input = ClaimInput
    final_output = VerificationOutput
    tools = [search_academic, verify_source]

class Agent(Module):
    branches = [FactCheckBranch]
    # Agent calls: branch(name="FactCheckBranch", claim="...")
```

### Deep analysis

```python
class AnalysisBranch(Module):
    """Perform in-depth analysis with specialized tools."""
    model = "anthropic/claude-opus-4-5"
    initial_input = AnalysisInput
    final_output = AnalysisOutput
    tools = [advanced_search, data_mining]
    max_steps = 20

class Agent(Module):
    branches = [AnalysisBranch]
    # Use for complex analysis without cluttering main agent
```

### Parallel processing (map-reduce)

```python
class ItemProcessorBranch(Module):
    """Process one item."""
    initial_input = ItemInput
    final_output = ItemOutput

class Orchestrator(Module):
    branches = [ItemProcessorBranch]
    # For each item: branch(name="ItemProcessorBranch", ...)
    # Collect results and aggregate
```

### Summarization

```python
class SummaryBranch(Module):
    """Summarize the conversation."""
    model = "anthropic/claude-haiku-4-5"
    final_output = SummaryOutput
    max_steps = 3
    # No initial_input - uses inherited history

class Agent(Module):
    branches = [SummaryBranch]
    # Agent calls: branch(name="SummaryBranch")
```

## Best practices

1. **Keep branch scope focused**: Each branch should have one clear purpose
2. **Use appropriate models**: Branches can use different (often cheaper/faster) models
3. **Define clear final_output**: Well-structured outputs make results easier to process
4. **Avoid deep nesting**: Limit branch depth to 1-2 levels
5. **Use merge strategies wisely**: `end_result` for simple tasks, `summarize` for complex workflows
6. **Handle errors gracefully**: Catch `BranchError` in manual branching
7. **Consider cost**: Each branch adds API calls - use `max_steps` to limit execution

## Limitations

- Branches run synchronously - parent waits for each branch to complete
- No shared state between parallel branches
- Branch history is copied, not shared - changes in one branch don't affect others
- Nested branches inherit the full chain of context and tools - can become unwieldy
