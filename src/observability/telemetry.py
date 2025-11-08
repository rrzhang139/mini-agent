"""Observability and telemetry for the agent."""
import logging
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

# In-memory trace storage (for debugging)
_trace_log: List[Dict[str, Any]] = []


def log_react_step(step_type: str, content: str, metadata: Optional[Dict] = None):
    """Log a ReAct step (Thought, Action, Observation, Final Answer)."""
    step = {
        "timestamp": datetime.now().isoformat(),
        "type": step_type,
        "content": content,
        "metadata": metadata or {}
    }
    _trace_log.append(step)

    # Format for console output
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
    logger.info(
        f"🔨 Tool: {tool_name}({arguments}) → {str(result)[:100]} ({duration_ms:.1f}ms)")


def log_node_entry(node_name: str, state: Dict):
    """Log when entering a node."""
    logger.info(f"📍 Entering node: {node_name}")
    logger.debug(f"  State: query='{state.get('query', '')[:50]}...', "
                 f"iteration={state.get('iteration_count', 0)}")


def log_node_exit(node_name: str, state: Dict):
    """Log when exiting a node."""
    logger.info(f"📍 Exiting node: {node_name}")
    if state.get("final_answer"):
        logger.info(f"  Final answer: {state['final_answer'][:100]}...")


def get_trace() -> List[Dict[str, Any]]:
    """Get the full trace log."""
    return _trace_log.copy()


def clear_trace():
    """Clear the trace log."""
    _trace_log.clear()


def format_trace_summary() -> str:
    """Format a human-readable trace summary."""
    if not _trace_log:
        return "No trace data"

    lines = ["\n=== Agent Execution Trace ==="]
    for step in _trace_log:
        timestamp = step["timestamp"].split("T")[1].split(".")[0]
        lines.append(
            f"[{timestamp}] {step['type'].upper()}: {step['content'][:150]}")

    return "\n".join(lines)
