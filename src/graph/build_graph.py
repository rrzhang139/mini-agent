"""Build the LangGraph agent graph."""
from langgraph.graph import StateGraph, END, START
from src.graph.state import State
from src.graph.nodes import (
    finalize_node, tool_node,
    initialize_node, guard_node
)


def build_graph():
    """Build and return the compiled agent graph."""
    graph = StateGraph(State)

    graph.add_node("finalize", finalize_node)
    graph.add_node("tool", tool_node)
    graph.add_node("initialize", initialize_node)
    graph.add_node("guard", guard_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "guard")
    graph.add_edge("guard", "tool")
    graph.add_edge("tool", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()
    print("Graph built successfully!")
    print(f"Nodes: {list(graph.nodes.keys())}")
