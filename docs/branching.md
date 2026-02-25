---
title: Branching
nav_order: 3
---

# Branching

Branching allows a module to spawn sub-agents that run independently and return results to the parent. Branches are specialized modules that inherit context from their parent and add focused capabilities for specific tasks.

## What is branching?

A branch is a standard `Module` subclass that runs as a sub-agent within a parent module. When you register branches in a parent module, the parent can spawn these sub-agents to handle specialized tasks. Each branch:

- Inherits the parent's conversation history and tools
- Has its own system prompt and can define additional tools
- Returns structured results back to the parent
- Can use a different model than the parent

Branches are useful when you need to delegate specialized work, run parallel processing, or isolate complex tasks from the main agent.

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

## Defining a branch

A branch is a `Module` subclass with two key attributes:

```python
from acorn import Module, tool
from pydantic import BaseModel, Field

class ClaimInput(BaseModel):
    claim: str = Field(description="The claim to verify")

class VerificationOutput(BaseModel):
    verified: bool = Field(description="Whether the claim is verified")
    evidence: list[str] = Field(description="Supporting evidence")

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

**Key attributes:**

- `initial_input`: Defines what parameters the branch accepts when called (optional)
- `final_output`: Required - must be a Pydantic model that defines the return type
- `tools`: Branch-specific tools that augment the parent's tools
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

## Calling branches

When you register branches, Acorn automatically generates a `branch()` tool that the parent agent can call.

### Discovery mode

Call `branch()` with no arguments to list available branches:

```python
# Agent calls: branch()
# Returns XML listing available branches and their input schemas
```

### Execution mode

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

## Branch inheritance

When a branch is spawned, it inherits from its parent:

| Aspect | Inheritance Behavior |
|--------|---------------------|
| **System prompt** | Branch uses its own `system_prompt` (not inherited) |
| **Tools** | Parent tools + branch tools + `call_parent_tool()` + `__finish__` |
| **History** | Full parent conversation history (copied, not referenced) |
| **Context** | Branch sees everything the parent has discussed |
| **Output** | Must call `__finish__()` with `final_output` schema |

## Accessing parent tools

Branches automatically receive a `call_parent_tool()` tool that provides access to the parent's tools.

**Discovery mode:**

```python
# Branch calls: call_parent_tool()
# Returns JSON listing parent tools
```

**Execution mode:**

```python
# Branch calls: call_parent_tool(name="search_web", query="Python history")
# Executes parent's search_web tool
# Returns the tool's result
```

**Note**: `call_parent_tool()` filters out `__finish__` and `branch` from the parent's tool list - branches only access regular parent tools.

## Merge strategies

Merge strategies control how branch results are returned to the parent.

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

## Manual branching

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

## Error handling

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

## Next Steps

- See [Module](module.md) for details on module configuration
- Check [Getting Started](getting-started.md) for complete examples
- Explore [examples/](../examples/) for real implementations