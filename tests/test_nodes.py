"""Tests for RAG functionality."""
import pytest
from src.graph.nodes import rag_node
from src.graph.state import State, build_initial_state


def test_rag_node_returns_complete_state():
    """Test that rag_node returns state with all expected fields."""
    state: State = build_initial_state(
        "What is the relocation allowance amount?")

    result = rag_node(state)

    assert "retrieved_chunks" in result
    assert "final_answer" in result
    assert "citations" in result

    assert len(result["retrieved_chunks"]) > 0
    assert result["final_answer"] is not None
    assert len(result["final_answer"]) > 0
    assert len(result["citations"]) > 0


def test_rag_node_answer_contains_key_information():
    """Test that the answer contains expected information for known queries."""
    state: State = build_initial_state(
        "What is the relocation allowance amount?")

    result = rag_node(state)

    answer_lower = result["final_answer"].lower()
    assert "5000" in answer_lower or "$5,000" in answer_lower or "5000" in answer_lower


def test_rag_node_citations_match_retrieved_chunks():
    """Test that citations match the sources of retrieved chunks."""
    state: State = build_initial_state(
        "What is the relocation allowance amount?")

    result = rag_node(state)

    chunk_sources = set(chunk["source"]
                        for chunk in result["retrieved_chunks"])
    citation_sources = set(result["citations"])

    assert citation_sources == chunk_sources


def test_tool_node_calculator():
    """Test that tool_node can execute calculator tool."""
    from src.graph.nodes import tool_node
    from src.graph.state import State

    state: State = build_initial_state(
        "What is 10 * 5?")

    result = tool_node(state)

    assert "final_answer" in result
    assert result["final_answer"] is not None
    assert len(result["final_answer"]) > 0
    # The answer should contain "50" somewhere
    assert "50" in result["final_answer"] or "fifty" in result["final_answer"].lower(
    )


def test_tool_node_returns_state():
    """Test that tool_node returns complete state."""
    from src.graph.nodes import tool_node
    from src.graph.state import State

    state: State = build_initial_state(
        "Calculate 2 + 2")

    result = tool_node(state)

    assert "final_answer" in result
    assert result["final_answer"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
