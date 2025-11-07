"""Configuration management for the agent."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
INDEX_DIR = DATA_DIR / "index"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_DEFAULT_CHANNEL = os.getenv("SLACK_DEFAULT_CHANNEL", "#general")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K = int(os.getenv("TOP_K", "5"))

MAX_NODE_ITERATIONS = int(os.getenv("MAX_NODE_ITERATIONS", "10"))
MAX_TOOL_HOPS = int(os.getenv("MAX_TOOL_HOPS", "3"))
TOOL_TIMEOUT_SECONDS = int(os.getenv("TOOL_TIMEOUT_SECONDS", "30"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

CORPUS_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
