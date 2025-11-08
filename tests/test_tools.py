"""Unit tests for agent tools."""
import pytest
import time
from src.tools.calculator import safe_calculate
from src.tools.slack import send_message, get_messages, list_channels


def test_safe_calculate_basic_operations():
    """Test basic arithmetic operations."""
    assert safe_calculate("1 + 1") == 2
    assert safe_calculate("5 - 2") == 3
    assert safe_calculate("3 * 4") == 12
    assert safe_calculate("10 / 2") == 5
    assert safe_calculate("2 ** 3") == 8


def test_safe_calculate_order_of_operations():
    """Test correct order of operations."""
    assert safe_calculate("2 + 3 * 4") == 14
    assert safe_calculate("(2 + 3) * 4") == 20
    assert safe_calculate("10 - 4 / 2") == 8


def test_safe_calculate_unary_minus():
    """Test unary minus operator."""
    assert safe_calculate("-5") == -5
    assert safe_calculate("-(5 + 2)") == -7


def test_safe_calculate_invalid_expression():
    """Test that invalid expressions raise ValueError."""
    with pytest.raises(ValueError, match="Invalid expression"):
        safe_calculate("1 + a")
    with pytest.raises(ValueError, match="Unsupported operation"):
        safe_calculate("1 & 2")


def test_safe_calculate_result_out_of_range():
    """Test that results outside the safe range raise ValueError."""
    with pytest.raises(ValueError, match="Result out of allowed range"):
        safe_calculate("10 ** 10")

# Slack integration tests


def test_list_slack_channels():
    """Test listing Slack channels."""
    channels = list_channels()
    assert isinstance(channels, list), "Should return a list of channels"


def test_create_slack_message():
    """Test creating slack messages and reading them back."""
    channel = "new-channel"
    test_message = f"Test message at {time.time()}"

    # Send message
    message_ts = send_message(test_message, channel)
    assert message_ts is not None, "Message should be sent successfully"

    # Wait a moment for Slack to process
    time.sleep(1)

    # Read messages from channel
    messages = get_messages(channel, limit=5)
    assert len(messages) > 0, "Should retrieve at least one message"

    # Find our test message
    found = False
    for msg in messages:
        if msg.get("text") == test_message:
            found = True
            assert msg.get(
                "ts") == message_ts, "Message timestamp should match"
            break

    assert found, f"Test message '{test_message}' should be found in channel"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
