"""CLI interface for the agent."""
import sys
from src.graph.build_graph import build_graph
from src.graph.state import build_initial_state
from src.config import LOG_LEVEL
from src.observability.telemetry import format_trace_summary, clear_trace
from src.observability.logging_config import configure_logging

# Configure logging (suppresses HTTP request logs)
configure_logging(LOG_LEVEL)


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.app 'your question here'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    clear_trace()  # Clear trace for fresh run

    # Build graph
    graph = build_graph()

    # Initialize state
    initial_state = build_initial_state(query)

    # Run the graph
    result = graph.invoke(initial_state)

    # Print trace summary
    print(format_trace_summary())

    # Print results
    print(f"\n{'='*60}")
    print(f"Answer: {result['final_answer']}")
    if result.get('citations'):
        print(f"\nSources: {', '.join(result['citations'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
