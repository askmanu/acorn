"""Quick test to verify metadata support."""

from pydantic import BaseModel
from acorn import Module
from unittest.mock import patch, MagicMock


class TextInput(BaseModel):
    text: str


class SentimentOutput(BaseModel):
    sentiment: str
    confidence: float


class SentimentClassifier(Module):
    """Classify the sentiment of the input text."""

    model = "gpt-4"
    initial_input = TextInput
    final_output = SentimentOutput
    metadata = {
        "user_id": "test_user_123",
        "session_id": "session_abc",
        "custom_field": "test_value"
    }


def test_metadata_passed_to_litellm():
    """Test that metadata is correctly passed to LiteLLM."""

    # Create a mock response that LiteLLM would return
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = None

    # Create proper tool call mock
    tool_call_mock = MagicMock()
    tool_call_mock.id = "call_123"
    tool_call_mock.function.name = "__finish__"
    tool_call_mock.function.arguments = '{"sentiment": "positive", "confidence": 0.95}'

    mock_response.choices[0].message.tool_calls = [tool_call_mock]
    mock_response.choices[0].finish_reason = "tool_calls"

    # Patch litellm.completion
    with patch('litellm.completion', return_value=mock_response) as mock_completion:
        classifier = SentimentClassifier()
        result = classifier(text="This is great!")

        # Verify the call was made
        assert mock_completion.called

        # Get the kwargs passed to litellm.completion
        call_kwargs = mock_completion.call_args[1]

        # Verify metadata was included
        assert "metadata" in call_kwargs
        assert call_kwargs["metadata"] == {
            "user_id": "test_user_123",
            "session_id": "session_abc",
            "custom_field": "test_value"
        }

        # Verify the result is correct
        assert result.sentiment == "positive"
        assert result.confidence == 0.95


def test_module_without_metadata():
    """Test that modules without metadata work as before."""

    class SimpleClassifier(Module):
        """Classify sentiment."""
        model = "gpt-4"
        initial_input = TextInput
        final_output = SentimentOutput
        # No metadata attribute

    # Create a mock response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message = MagicMock()
    mock_response.choices[0].message.content = None

    # Create proper tool call mock
    tool_call_mock = MagicMock()
    tool_call_mock.id = "call_456"
    tool_call_mock.function.name = "__finish__"
    tool_call_mock.function.arguments = '{"sentiment": "negative", "confidence": 0.8}'

    mock_response.choices[0].message.tool_calls = [tool_call_mock]
    mock_response.choices[0].finish_reason = "tool_calls"

    # Patch litellm.completion
    with patch('litellm.completion', return_value=mock_response) as mock_completion:
        classifier = SimpleClassifier()
        result = classifier(text="This is bad!")

        # Verify the call was made
        assert mock_completion.called

        # Get the kwargs passed to litellm.completion
        call_kwargs = mock_completion.call_args[1]

        # Verify metadata is not in kwargs (since it's None, our code doesn't add it)
        assert "metadata" not in call_kwargs

        # Verify the result is correct
        assert result.sentiment == "negative"
        assert result.confidence == 0.8


if __name__ == "__main__":
    print("Running metadata tests...")
    test_metadata_passed_to_litellm()
    print("✓ Test 1 passed: Metadata is correctly passed to LiteLLM")

    test_module_without_metadata()
    print("✓ Test 2 passed: Modules without metadata work as before")

    print("\nAll tests passed! ✓")
