"""Centralized logging configuration."""
import logging


def configure_logging(level: str = "INFO"):
    """Configure logging with appropriate levels for different modules."""
    log_level = getattr(logging, level.upper())

    # Base configuration
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        force=True  # Override any existing configuration
    )

    # Suppress noisy HTTP logs from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)

    # Suppress OpenAI SDK internal logs (keep only errors)
    logging.getLogger("openai").setLevel(logging.WARNING)

    # Suppress FAISS loader logs
    logging.getLogger("faiss.loader").setLevel(logging.WARNING)

    # Keep our observability logs visible
    logging.getLogger("src.observability").setLevel(logging.INFO)
    logging.getLogger("src.graph.nodes").setLevel(logging.INFO)
    logging.getLogger("src.guards").setLevel(logging.INFO)
    # Set root logger to the desired level
    logging.getLogger().setLevel(log_level)
