"""Build the LangGraph agent graph."""
from langgraph.graph import StateGraph, END, START
from src.graph.state import State
from src.graph.nodes import (
    rag_node, finalize_node, router, tool_node,
    initialize_node, guard_node, guard_router
)


def build_graph():
    """Build and return the compiled agent graph."""
    graph = StateGraph(State)

    graph.add_node("rag", rag_node)
    graph.add_node("finalize", finalize_node)
    graph.add_node("tool", tool_node)
    graph.add_node("initialize", initialize_node)
    graph.add_node("guard", guard_node)

    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "guard")
    graph.add_conditional_edges("guard", guard_router)
    graph.add_conditional_edges("rag", router)
    graph.add_conditional_edges("tool", router)
    graph.add_edge("finalize", END)

    return graph.compile()


if __name__ == "__main__":
    graph = build_graph()
    print("Graph built successfully!")
    print(f"Nodes: {list(graph.nodes.keys())}")
