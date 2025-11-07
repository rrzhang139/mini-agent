from typing import TypedDict, Dict, List, Optional, Any
from datetime import datetime


class ToolCall(TypedDict):
    name: str
    arguments: Dict
    result: Any
    timestamp: datetime


class State(TypedDict):
    query: str
    retrieved_chunks: List[Dict]
    final_answer: Optional[str]
    citations: Optional[List[str]]
    messages: List[Dict]
    tool_calls: Optional[List[ToolCall]]
    iteration_count: int = 0


def build_initial_state(query: str) -> State:
    state = State(
        query=query,
        retrieved_chunks=[],
        final_answer="",
        citations=[],
        messages=[],
        tool_calls=[],
        iteration_count=0)
    return state
