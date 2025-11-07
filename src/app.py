"""CLI interface for the agent."""
import sys
from src.graph.build_graph import build_graph
from src.graph.state import build_initial_state


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.app 'your question here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])

    # Build graph
    graph = build_graph()

    # Initialize state
    initial_state = build_initial_state(query)

    # Run the graph
    result = graph.invoke(initial_state)

    # Print results
    print(f"\nAnswer: {result['final_answer']}")
    print(f"\nSources: {', '.join(result['citations'])}")


if __name__ == "__main__":
    main()
