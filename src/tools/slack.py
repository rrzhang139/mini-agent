from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
from src.config import SLACK_BOT_TOKEN, SLACK_DEFAULT_CHANNEL

client = WebClient(token=SLACK_BOT_TOKEN)


def _resolve_channel_id(channel: str) -> str:
    """Resolve channel name to channel ID.

    If channel is already an ID (starts with C, G, or D), return as-is.
    Otherwise, look up by name.
    """
    # If it's already an ID format, return as-is
    if channel.startswith(("C", "G", "D")):
        return channel

    # Remove # if present
    channel_name = channel.lstrip("#")

    # Look up channel by name
    try:
        response = client.conversations_list()
        for ch in response["channels"]:
            if ch["name"] == channel_name:
                return ch["id"]
        raise ValueError(f"Channel '{channel}' not found")
    except SlackApiError as e:
        raise ValueError(
            f"Failed to resolve channel '{channel}': {e.response.get('error')}")


def send_message(text: str, channel: str = None):
    """Send a message to a Slack channel.

    Args:
        text: Message text to send
        channel: Channel name (with or without #) or channel ID. Defaults to SLACK_DEFAULT_CHANNEL.
    """
    try:
        channel_id = _resolve_channel_id(channel or SLACK_DEFAULT_CHANNEL)
        response = client.chat_postMessage(
            channel=channel_id,
            text=text
        )
        return response["ts"]
    except (SlackApiError, ValueError) as e:
        print(f"Slack API Error: {e}")
        return None


def list_channels(types: str = "public_channel,private_channel"):
    """List all channels the bot has access to."""
    try:
        response = client.conversations_list(types=types)
        return response["channels"]
    except SlackApiError as e:
        print(f"Slack API Error: {e.response['error']}")
        return []


def get_messages(channel: str = None, limit: int = 10):
    """Get recent messages from a Slack channel.

    Args:
        channel: Channel name (with or without #) or channel ID. Defaults to SLACK_DEFAULT_CHANNEL.
        limit: Maximum number of messages to retrieve
    """
    try:
        channel_id = _resolve_channel_id(channel or SLACK_DEFAULT_CHANNEL)
        response = client.conversations_history(
            channel=channel_id,
            limit=limit
        )
        return response["messages"]
    except (SlackApiError, ValueError) as e:
        print(f"Slack API Error: {e}")
        return []
