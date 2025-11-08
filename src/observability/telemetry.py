"""Observability and telemetry for the agent."""
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class RecordType(str, Enum):
    """Types of trace records."""
    REACT_STEP = "react_step"
    TOOL_CALL = "tool_call"
    NODE_ENTRY = "node_entry"
    NODE_EXIT = "node_exit"


@dataclass
class TraceRecord:
    """Base class for trace records."""
    timestamp: datetime

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to dictionary for serialization."""
        record_type = getattr(self, 'record_type', None)
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": record_type.value if record_type else "unknown",
            **{k: v for k, v in self.__dict__.items() if k not in ("timestamp", "record_type")}
        }


@dataclass
class ReactStepRecord(TraceRecord):
    """Record for ReAct steps (Thought, Action, Observation, Final Answer)."""
    step_type: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    record_type: RecordType = field(default=RecordType.REACT_STEP, init=False)


@dataclass
class ToolCallRecord(TraceRecord):
    """Record for tool calls."""
    tool_name: str
    arguments: Dict[str, Any]
    result: Any
    duration_ms: float
    record_type: RecordType = field(default=RecordType.TOOL_CALL, init=False)


@dataclass
class NodeEntryRecord(TraceRecord):
    """Record for node entry."""
    node_name: str
    query: str
    iteration_count: int
    record_type: RecordType = field(default=RecordType.NODE_ENTRY, init=False)


@dataclass
class NodeExitRecord(TraceRecord):
    """Record for node exit."""
    node_name: str
    final_answer: Optional[str] = None
    record_type: RecordType = field(default=RecordType.NODE_EXIT, init=False)


# In-memory trace storage
_trace_log: List[TraceRecord] = []


def log_react_step(step_type: str, content: str, metadata: Optional[Dict] = None):
    """Log a ReAct step (Thought, Action, Observation, Final Answer)."""
    record = ReactStepRecord(
        timestamp=datetime.now(),
        step_type=step_type,
        content=content,
        metadata=metadata or {}
    )
    _trace_log.append(record)

    prefix = {
        "thought": "💭",
        "action": "🔧",
        "observation": "👁️",
        "final_answer": "✅"
    }.get(step_type.lower(), "📝")

    logger.info(f"{prefix} {step_type.upper()}: {content[:200]}")
    if metadata:
        logger.debug(f"  Metadata: {metadata}")


def log_tool_call(tool_name: str, arguments: Dict, result: Any, duration_ms: float):
    """Log a tool call with timing."""
    record = ToolCallRecord(
        timestamp=datetime.now(),
        tool_name=tool_name,
        arguments=arguments,
        result=result,
        duration_ms=duration_ms
    )
    _trace_log.append(record)
    logger.info(
        f"🔨 Tool: {tool_name}({arguments}) → {str(result)[:100]} ({duration_ms:.1f}ms)")


def log_node_entry(node_name: str, state: Dict):
    """Log when entering a node."""
    record = NodeEntryRecord(
        timestamp=datetime.now(),
        node_name=node_name,
        query=state.get("query", "")[:50],
        iteration_count=state.get("iteration_count", 0)
    )
    _trace_log.append(record)
    logger.info(f"📍 Entering node: {node_name}")
    logger.debug(f"  State: query='{state.get('query', '')[:50]}...', "
                 f"iteration={state.get('iteration_count', 0)}")


def log_node_exit(node_name: str, state: Dict):
    """Log when exiting a node."""
    record = NodeExitRecord(
        timestamp=datetime.now(),
        node_name=node_name,
        final_answer=state.get("final_answer")
    )
    _trace_log.append(record)
    logger.info(f"📍 Exiting node: {node_name}")
    if state.get("final_answer"):
        logger.info(f"  Final answer: {state['final_answer'][:100]}...")


def get_trace() -> List[TraceRecord]:
    """Get the full trace log."""
    return _trace_log.copy()


def get_trace_dicts() -> List[Dict[str, Any]]:
    """Get trace log as list of dictionaries (for backward compatibility)."""
    return [record.to_dict() for record in _trace_log]


def get_react_steps() -> List[ReactStepRecord]:
    """Get all ReAct step records."""
    return [r for r in _trace_log if isinstance(r, ReactStepRecord)]


def get_tool_calls() -> List[ToolCallRecord]:
    """Get all tool call records."""
    return [r for r in _trace_log if isinstance(r, ToolCallRecord)]


def get_node_entries() -> List[NodeEntryRecord]:
    """Get all node entry records."""
    return [r for r in _trace_log if isinstance(r, NodeEntryRecord)]


def clear_trace():
    """Clear the trace log."""
    _trace_log.clear()


def format_trace_summary() -> str:
    """Format a human-readable trace summary."""
    if not _trace_log:
        return "No trace data"

    lines = ["\n=== Agent Execution Trace ==="]
    for record in _trace_log:
        timestamp = record.timestamp.strftime("%H:%M:%S")
        if isinstance(record, ReactStepRecord):
            lines.append(
                f"[{timestamp}] {record.step_type.upper()}: {record.content[:150]}")
        elif isinstance(record, ToolCallRecord):
            lines.append(
                f"[{timestamp}] TOOL: {record.tool_name}({record.arguments})")
        elif isinstance(record, NodeEntryRecord):
            lines.append(f"[{timestamp}] ENTER: {record.node_name}")
        elif isinstance(record, NodeExitRecord):
            lines.append(f"[{timestamp}] EXIT: {record.node_name}")

    return "\n".join(lines)
