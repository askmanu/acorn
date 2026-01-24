# Branching

Branching allows a module to spawn sub-agents that run to completion and return results to the main "trunk" of execution. Branches **extend** the parent module rather than replacing it.

## Branch Inheritance Model

When a branch is spawned, it inherits from its parent and adds its own capabilities:

| Aspect | Behavior |
|--------|----------|
| **System prompt** | Parent's prompt + branch's docstring (appended) |
| **Tools** | Parent's tools + branch's tools + `__finish__` |
| **History** | Full parent history inherited at branch point |
| **Trigger** | Tool call with branch ID only (no arguments) |
| **Output** | Structured `final_output` (branch must define Pydantic schema) |

```
Main Module (trunk)
    │
    ├── Step 1: search_web()
    │
    ├── Step 2: analyze_data()
    │
    ├── Step 3: fact_check() ──────────┐
    │                                  │
    │   ┌──────────────────────────────▼──────────────────────────────┐
    │   │  FactCheckBranch                                            │
    │   │    System: Parent prompt + "Verify claims discussed..."     │
    │   │    Tools: [search, calculate, deep_search, verify, __finish__] │
    │   │    History: Full parent history                             │
    │   │                                                             │
    │   │    ├── Step 1: deep_search()                                │
    │   │    ├── Step 2: verify_source()                              │
    │   │    └── Step 3: __finish__(is_true=True, evidence=[...])     │
    │   └──────────────────────────────┬──────────────────────────────┘
    │                                  │
    ├── Step 4: (branch result injected as tool result) ◄─────────────┘
    │
    └── Step 5: __finish__(answer="...")
```

The branch runs synchronously - the main module waits for it to complete before continuing.

---

## Defining a Branch Module

A branch is a module that **extends** its parent. It uses its docstring as additional system prompt instructions:

```python
from acorn import module, tool
from pydantic import BaseModel, Field

@tool
def deep_search(query: str) -> list[str]:
    """Perform deep web search with multiple sources."""
    return [...]

@tool
def verify_source(url: str) -> dict:
    """Verify the credibility of a source."""
    return {"credible": True, "score": 0.9}


class FactCheckOutput(BaseModel):
    is_true: bool = Field(description="Whether the claim is verified")
    evidence: list[str] = Field(description="Supporting evidence")
    confidence: float = Field(description="Confidence score 0-1")


class FactCheckBranch(module):
    """
    Verify the claims discussed in the conversation.
    Search for evidence and assess truthfulness.
    Use multiple sources to corroborate information.
    """
    # Docstring = additional prompt appended to parent's system_prompt

    tools = [deep_search, verify_source]  # Added to parent's tools
    final_output = FactCheckOutput         # Required - defines __finish__ schema

    # No initial_input - context comes from inherited history
    # No model - inherits from parent (can override if needed)

    max_steps = 8  # Can override parent's settings
    temperature = 0.2  # Lower temperature for verification
```

### Key Points

- **Docstring as prompt**: The branch's docstring is appended to the parent's system prompt
- **No `initial_input`**: Branches don't take arguments - they inherit full conversation history
- **`final_output` required**: Must define what `__finish__` returns
- **Tools are additive**: Branch tools are added to parent's tools
- **Settings can override**: Model, temperature, max_steps, etc. can be customized

---

## Registering Branches

In the parent module, register branches in the `branches` dict:

```python
class ResearchAgent(module):
    """You are a research assistant. Answer questions thoroughly."""

    model = "anthropic/claude-sonnet-4-5-20250514"
    tools = [search_web, calculate]

    branches = {
        "fact_check": FactCheckBranch,
        "deep_analysis": AnalysisBranch,
    }

    final_output = ResearchOutput
```

### Auto-Generated Branch Tools

For each branch, acorn generates a simple tool with no parameters:

| Branch Key | Generated Tool |
|------------|----------------|
| `"fact_check": FactCheckBranch` | `fact_check() -> FactCheckOutput` |

The model sees these as regular tools:

```
Available tools:
- search_web(query: str, max_results: int) -> list[dict]
- calculate(expression: str) -> float
- fact_check() -> {is_true: bool, evidence: list[str], confidence: float}
- deep_analysis() -> {summary: str, key_findings: list[str]}
- __finish__(...)
```

The tool description comes from the branch's docstring (first paragraph).

---

## Branch Execution

When the model calls a branch tool:

1. **Branch initialized** with inherited context:
   - System prompt = parent's prompt + branch's docstring
   - Tools = parent's tools + branch's tools + `__finish__`
   - History = copy of parent's full history

2. **Branch runs** its own agentic loop until `__finish__` is called

3. **Result returned** to parent as a tool result (serialized `final_output`)

4. **Parent continues** with the branch result in its context

### Example Flow

```python
# Parent is running, model decides to fact-check a claim
# Model calls: fact_check()

# 1. FactCheckBranch starts with:
#    System: "You are a research assistant..." + "Verify the claims discussed..."
#    Tools: [search_web, calculate, deep_search, verify_source, __finish__]
#    History: [all parent messages so far]

# 2. Branch runs, making tool calls:
#    - deep_search("python creation date 1991")
#    - verify_source("https://python.org/history")
#    - __finish__(is_true=True, evidence=["..."], confidence=0.95)

# 3. Result returned to parent as tool result:
#    {"is_true": true, "evidence": ["..."], "confidence": 0.95}

# 4. Parent continues, seeing fact_check() returned verification data
```

---

## Custom Branch Configuration

### Override Model

Use a different (often cheaper/faster) model for specific branches:

```python
class SummaryBranch(module):
    """Summarize the discussion concisely."""

    model = "anthropic/claude-haiku"  # Faster model for simple task
    final_output = SummaryOutput
    max_steps = 3
```

### Custom Tool Description

Override the auto-generated description:

```python
branches = {
    "fact_check": {
        "module": FactCheckBranch,
        "description": "Verify factual claims with evidence from multiple sources",
    },
}
```

---

## Manual Branching

Spawn branches programmatically in `on_step` for cases where you need custom logic:

```python
def on_step(self, step):
    # Check if we need to branch based on tool results
    for result in step.tool_results:
        if result.name == "analyze_data" and result.output.get("needs_verification"):
            # Spawn verification branch manually
            verification = self.branch(FactCheckBranch)

            # Inject result into history for next iteration
            self.history.append({
                "role": "user",
                "content": f"Verification result: {verification.model_dump_json()}"
            })

    return step
```

### When to Use Manual vs Declarative

| Use Case | Approach |
|----------|----------|
| Model decides when to branch | Declarative (`branches = {}`) |
| Conditional branching based on results | Manual (`self.branch()` in `on_step`) |
| Branch with modified history | Manual |
| Simple "spawn specialist" pattern | Declarative |

---

## Nested Branches

Branches can define their own branches (nested branching):

```python
class DeepResearchBranch(module):
    """Perform deep research on a topic."""

    branches = {
        "verify_fact": FactCheckBranch,
    }
    # DeepResearchBranch can call verify_fact()
    # verify_fact will inherit from DeepResearchBranch (which inherited from root)
```

Nested branches inherit through the full chain:
- `verify_fact` gets DeepResearchBranch's accumulated system prompt + its own docstring
- Tools accumulate through all levels

Be cautious with nesting depth - each level adds latency and cost.

---

## Branch Errors

If a branch fails (forced output fails to parse after retries, etc.):

**Declarative branch**: Error returned to parent model as tool result. Model can acknowledge and continue.

```
fact_check() returned error: ParseError - could not validate output after 2 retries
```

**Manual branch**: Exception raised. Handle in `on_step`:

```python
def on_step(self, step):
    try:
        result = self.branch(RiskyBranch)
    except BranchError as e:
        # Handle failure - inject error into context
        self.history.append({
            "role": "user",
            "content": f"Verification failed: {e}"
        })
    return step
```

---

## Complete Example

```python
from acorn import module, tool
from pydantic import BaseModel, Field

# --- Tools ---

@tool
def search_web(query: str) -> list[str]:
    """Search the web for information."""
    return ["result1", "result2"]

@tool
def deep_search(query: str) -> list[str]:
    """Deep search with academic sources."""
    return ["academic result 1", "academic result 2"]

@tool
def verify_source(url: str) -> dict:
    """Check source credibility."""
    return {"credible": True}

# --- Branch Module ---

class VerificationOutput(BaseModel):
    is_true: bool
    evidence: list[str]
    confidence: float

class FactCheckBranch(module):
    """
    Verify the claims discussed in the conversation.
    Search for evidence and assess truthfulness.
    """

    tools = [deep_search, verify_source]
    final_output = VerificationOutput
    max_steps = 8
    temperature = 0.2

# --- Main Module ---

class ResearchOutput(BaseModel):
    answer: str
    sources: list[str]
    verified: bool

class ResearchAgent(module):
    """
    You are a research assistant. Answer questions thoroughly.
    Use fact_check when you need to verify important claims.
    """

    model = "anthropic/claude-sonnet-4-5-20250514"
    temperature = 0.7
    max_steps = 20

    tools = [search_web]

    branches = {
        "fact_check": FactCheckBranch,
    }

    final_output = ResearchOutput

    def on_step(self, step):
        # Log branch calls
        for call in step.tool_calls:
            if call.name in self.branches:
                print(f"Branch spawned: {call.name}")
        return step


# --- Usage ---

agent = ResearchAgent()
result = agent(question="Was Python created in 1991?")

# The model might:
# 1. search_web("Python creation date")
# 2. fact_check()  <- spawns branch with inherited context
#    Branch runs with full history, verifies the claim
#    Returns: {"is_true": True, "evidence": [...], "confidence": 0.95}
# 3. __finish__(answer="Yes, Python was created in 1991", verified=True, ...)
```

---

## Summary

| Old Model | New Model |
|-----------|-----------|
| Branch has `initial_input` | No `initial_input` - inherits history |
| Branch has independent system prompt | Branch docstring appended to parent's |
| Branch has independent tools | Branch tools added to parent's |
| Branch called with arguments | Branch called with no arguments |
| Branch starts fresh | Branch inherits full context |
