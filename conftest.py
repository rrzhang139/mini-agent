"""Pytest configuration for test logging."""
from src.config import LOG_LEVEL
from src.observability.logging_config import configure_logging

configure_logging(LOG_LEVEL)
