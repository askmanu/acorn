# Data Serialization

Acorn uses XML internally to communicate structured data with LLMs. Users work with Pydantic models and Python types - they never write XML directly.

## Design Philosophy

- **User-facing**: Pydantic models, Python type hints
- **LLM-facing**: XML for structured data, native tool schemas for tools
- **System prompts**: Markdown (no XML)

This separation keeps the user API clean while using XML's strengths for LLM communication (clear structure, natural language friendly, explicit boundaries).

---

## Why XML for LLM Communication

1. **Clear boundaries**: Tags explicitly mark where data begins and ends
2. **Nested structure**: Natural representation of complex data
3. **LLM familiarity**: Models are trained on vast amounts of XML/HTML
4. **No escaping issues**: Unlike JSON, natural language in XML doesn't need escaping
5. **Self-documenting**: Tag names describe the content

---

## When XML is Used

| Scenario | Format |
|----------|--------|
| System prompt | Markdown (no XML) |
| User input data → model | XML |
| Structured model response | XML (parsed to Pydantic) |
| Tool definitions | Native API schema (JSON) |
| Tool parameters | Native API format |
| Tool results | Native API format |

XML is primarily for:
1. Initial input data sent to the model
2. Structured responses from the model (non-tool responses)

---

## Input Serialization

When the module is called, `initial_input` is serialized to XML and included in the first user message.

### Pydantic to XML

```python
# User defines:
class QuestionInput(BaseModel):
    question: str = Field(description="The question to answer")
    context: str | None = Field(default=None, description="Optional context")

# User calls:
agent(question="What is Python?", context="Programming languages")

# Acorn generates (sent to model):
```
```xml
<input>
  <question>What is Python?</question>
  <context>Programming languages</context>
</input>
```

### Nested Structures

```python
class Source(BaseModel):
    title: str
    url: str

class ResearchInput(BaseModel):
    question: str
    sources: list[Source]
    max_depth: int = 3

# Called with:
agent(
    question="Compare frameworks",
    sources=[
        Source(title="React Docs", url="https://react.dev"),
        Source(title="Vue Docs", url="https://vuejs.org"),
    ]
)

# Serialized:
```
```xml
<input>
  <question>Compare frameworks</question>
  <sources>
    <source>
      <title>React Docs</title>
      <url>https://react.dev</url>
    </source>
    <source>
      <title>Vue Docs</title>
      <url>https://vuejs.org</url>
    </source>
  </sources>
  <max_depth>3</max_depth>
</input>
```

### Field Descriptions

Field descriptions can be included as XML comments or attributes (configurable):

```xml
<!-- As comments (default) -->
<input>
  <!-- The question to answer -->
  <question>What is Python?</question>
</input>

<!-- As attributes -->
<input>
  <question description="The question to answer">What is Python?</question>
</input>
```

---

## Output Parsing (Fallback Only)

Output is normally returned via the `__finish__` tool call (JSON arguments validated by Pydantic). XML output parsing is only used as a **fallback** when forcing termination at `max_steps` on providers that don't support `tool_choice`.

### Fallback System Prompt Injection

When XML fallback is needed, acorn adds instructions:

```
Respond with your answer in this XML format:
<output>
  <answer>Your answer here</answer>
  <confidence>0.0 to 1.0</confidence>
</output>
```

### XML to Pydantic

```xml
<!-- Model responds: -->
<output>
  <answer>Python is a programming language created in 1991.</answer>
  <confidence>0.95</confidence>
</output>
```

```python
# Parsed to:
AnswerOutput(
    answer="Python is a programming language created in 1991.",
    confidence=0.95
)
```

### Validation

1. Parse XML structure
2. Extract values
3. Convert internally to `__finish__` call format
4. Validate against Pydantic `final_output` model
5. If validation fails → error sent to model → retry

**Note**: This XML parsing path is rarely used. Normal output always flows through the `__finish__` tool call with JSON arguments.

---

## Context Injection

When using `step.add_to_context()`, content is wrapped in XML:

```python
def on_step(self, step):
    step.add_to_context({
        "analysis_result": {
            "summary": "Data shows trend",
            "confidence": 0.8
        }
    })
```

```xml
<!-- Added to next user message: -->
<context>
  <analysis_result>
    <summary>Data shows trend</summary>
    <confidence>0.8</confidence>
  </analysis_result>
</context>
```

Raw strings are wrapped simply:

```python
step.add_to_context("Remember to be concise")
```

```xml
<context>Remember to be concise</context>
```

---

## Tool Communication

Tools use **native API formats**, not XML. This leverages provider-specific optimizations.

### Tool Schema

Generated from decorated function, sent as JSON schema per API spec:

```python
@tool
def search(query: str, limit: int = 10) -> list[str]:
    """Search the web for information."""
    ...
```

```json
{
  "name": "search",
  "description": "Search the web for information.",
  "parameters": {
    "type": "object",
    "properties": {
      "query": {"type": "string"},
      "limit": {"type": "integer", "default": 10}
    },
    "required": ["query"]
  }
}
```

### Tool Calls and Results

Standard message format per provider (OpenAI/Anthropic style):

```python
# Model response (tool call)
{
    "role": "assistant",
    "tool_calls": [{
        "id": "call_123",
        "name": "search",
        "arguments": {"query": "Python history", "limit": 5}
    }]
}

# Tool result (sent back)
{
    "role": "tool",
    "tool_call_id": "call_123",
    "content": "[\"result1\", \"result2\", ...]"
}
```

---

## Serialization Options

Module-level configuration for XML serialization:

```python
class MyAgent(module):
    # Include field descriptions in XML
    xml_include_descriptions = True  # default: True

    # Description format: "comment" or "attribute"
    xml_description_format = "comment"  # default: "comment"

    # Root element names
    xml_input_root = "input"    # default: "input"
    xml_output_root = "output"  # default: "output"
    xml_context_root = "context"  # default: "context"
```

---

## Special Cases

### None Values

```python
# Field with None value:
agent(question="What is Python?", context=None)

# Option 1: Omit (default)
<input>
  <question>What is Python?</question>
</input>

# Option 2: Include as empty (configurable)
<input>
  <question>What is Python?</question>
  <context />
</input>
```

### Lists

```python
sources: list[str] = ["a", "b", "c"]
```

```xml
<sources>
  <item>a</item>
  <item>b</item>
  <item>c</item>
</sources>
```

### Dicts

```python
metadata: dict[str, str] = {"author": "John", "year": "2024"}
```

```xml
<metadata>
  <author>John</author>
  <year>2024</year>
</metadata>
```

### Large Text

Long text fields are preserved as-is (XML handles multiline naturally):

```xml
<document>
This is a very long document
with multiple lines
and paragraphs.

It can contain any text.
</document>
```

---

## Error Messages

When sending parse errors back to the model, acorn uses clear XML structure:

```xml
<error type="validation">
  <message>Output validation failed</message>
  <details>
    <field name="confidence">
      <expected>float between 0 and 1</expected>
      <received>very confident</received>
    </field>
  </details>
  <instruction>Please provide the output again in the correct format.</instruction>
</error>
```

---

## Implementation Notes

### XML Library

Use Python's `xml.etree.ElementTree` for simplicity, or `lxml` for advanced features.

### Escaping

Standard XML escaping for special characters:
- `<` → `&lt;`
- `>` → `&gt;`
- `&` → `&amp;`
- `"` → `&quot;` (in attributes)

### Pretty Printing

XML sent to models should be human-readable (indented) for better model comprehension:

```xml
<input>
  <question>What is Python?</question>
  <context>Programming</context>
</input>
```

Not:
```xml
<input><question>What is Python?</question><context>Programming</context></input>
```
