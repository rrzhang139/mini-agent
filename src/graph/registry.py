from src.tools.calculator import safe_calculate
from src.tools.calendar_mock import list_events, create_event, clear_events
from src.rag.retriever import load_index, load_chunks, retrieve
from src.tools.slack import send_message

_index = None
_chunks = None


def load_index_and_chunks():
    """Lazy load index and chunks on first use."""
    global _index, _chunks
    if _index is None:
        _index = load_index()
        _chunks = load_chunks()
    return _index, _chunks


def rag_search(query: str, k: int = 5):
    """Retrieve top-k relevant chunks for a query (RAG as a tool)."""
    index, chunks = load_index_and_chunks()
    results = retrieve(query, index, chunks, k=k)
    # Return lean payload for tool observation
    return [{"source": c["source"], "content": c["content"]} for c in results]


tools = {
    "safe_calculate": safe_calculate,
    "list_events": list_events,
    "create_event": create_event,
    "clear_events": clear_events,
    "rag_search": rag_search,
    "send_slack_message": send_message,
}
tool_definitions = [
    {
        "type": "function",
        "function": {
            "name": "safe_calculate",
            "description": "Safely calculate mathematical expressions. Supports +, -, *, /, **",
            "parameters": {
                "type": "object",
                "properties": {
                    "expr": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 3 * 4'"
                    }
                },
                "required": ["expr"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_events",
            "description": "List all events in the calendar",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "The start date of the events to list, e.g. '2025-11-01'"
                    },
                    "end_date": {
                        "type": "string",
                        "description": "The end date of the events to list, e.g. '2025-11-30'"
                    }
                },
                # Parameters are optional, so 'required' is omitted
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "create_event",
            "description": "Create a new event in the calendar",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "The title of the event, e.g. 'Meeting with John'"
                    },
                    "start_time": {
                        "type": "string",
                        "description": "The start time of the event, e.g. '2025-11-01T10:00'"
                    },
                    "duration_minutes": {
                        "type": "number",
                        "description": "The duration of the event in minutes, e.g. 60"
                    }
                },
                "required": ["title", "start_time", "duration_minutes"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "clear_events",
            "description": "Clear all events from the calendar",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "rag_search",
            "description": "Retrieve top-k relevant document chunks for a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Text to search in the document corpus"
                    },
                    "k": {
                        "type": "number",
                        "description": "Number of chunks to retrieve (default 5)"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "send_slack_message",
            "description": "Send a message to a Slack channel",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Message text to send"
                    },
                    "channel": {
                        "type": "string",
                        "description": "Channel name, e.g. '#new-channel' (optional)"
                    }
                },
                "required": ["text"]
            }
        }
    }
]


def load_tools():
    """Load tools and tool definitions on first use."""
    return tools, tool_definitions
