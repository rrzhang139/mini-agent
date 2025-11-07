"""Guards and policies for the agent."""
import re
import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

# Refuse patterns for queries that need grounding
REFUSE_PATTERNS = [
    (r"legal advice|lawyer|legal counsel|sue|lawsuit|legal opinion", "legal"),
    (r"medical advice|diagnose|prescription|treatment|doctor|symptom|disease|medicine", "medical"),
    (r"financial advice|investment|trading|stock pick|buy.*stock|sell.*stock", "financial"),
    (r"generate.*letter|write.*contract|draft.*document|create.*legal",
     "document_generation"),
]

# PII patterns to mask
PII_PATTERNS = [
    (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
    (r"\b\d{3}-\d{2}-\d{4}\b", "ssn"),  # SSN format: 123-45-6789
    (r"\b\d{3}\.\d{3}\.\d{4}\b", "phone"),  # Phone: 123.456.7890
    (r"\b\d{16}\b", "credit_card"),  # 16-digit credit card
]

# Sandbox directory for file operations
SANDBOX_DIR = PROJECT_ROOT / "data" / "sandbox"


def check_refuse_patterns(query: str) -> Tuple[bool, Optional[str]]:
    """
    Check if query matches refuse patterns (legal/medical advice, etc.).

    Args:
        query: User query to check

    Returns:
        (should_refuse, reason) tuple
    """
    query_lower = query.lower()
    for pattern, category in REFUSE_PATTERNS:
        if re.search(pattern, query_lower, re.IGNORECASE):
            logger.info(
                f"Query refused by pattern: {category} - '{query[:50]}...'")
            return True, category
    logger.debug(f"Query passed refuse pattern check: '{query[:50]}...'")
    return False, None


def mask_pii(text: str) -> str:
    """
    Mask PII (emails, SSNs, phone numbers, credit cards) in text.

    Args:
        text: Text that may contain PII

    Returns:
        Text with PII masked as [REDACTED_<type>]
    """
    masked = text
    masked_count = 0
    for pattern, pii_type in PII_PATTERNS:
        matches = re.findall(pattern, masked)
        if matches:
            masked_count += len(matches)
            masked = re.sub(pattern, f"[REDACTED_{pii_type.upper()}]", masked)

    if masked_count > 0:
        logger.info(f"Masked {masked_count} PII instances in text")

    return masked


def validate_file_path(file_path: str) -> Tuple[bool, Optional[str]]:
    """
    Validate that file path is within sandbox directory.

    Args:
        file_path: File path to validate

    Returns:
        (is_valid, error_message) tuple
    """
    try:
        # Resolve to absolute path
        requested_path = Path(file_path).resolve()
        sandbox_path = SANDBOX_DIR.resolve()

        # Check if path is within sandbox
        try:
            requested_path.relative_to(sandbox_path)
            logger.debug(f"File path validated: {file_path}")
            return True, None
        except ValueError:
            error_msg = f"Path {file_path} is outside sandbox directory {sandbox_path}"
            logger.warning(error_msg)
            return False, error_msg
    except Exception as e:
        error_msg = f"Invalid path: {str(e)}"
        logger.error(f"Path validation error: {error_msg}")
        return False, error_msg


def check_grounding_required(
    query: str,
    retrieved_chunks: List[Dict]
) -> Tuple[bool, Optional[str]]:
    """
    Check if answer requires grounding (citations from retrieved chunks).

    Args:
        query: User query
        retrieved_chunks: List of retrieved chunks
        grounding_required: Whether grounding is required for this query type

    Returns:
        (is_grounded, error_message) tuple
    """

    if not retrieved_chunks or len(retrieved_chunks) == 0:
        # Check if query is asking for factual information
        factual_patterns = [
            r"what is|how much|when|where|who|explain|describe",
            r"policy|procedure|allowance|rate|formula",
        ]
        query_lower = query.lower()
        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                error_msg = "Query requires grounding but no documents retrieved"
                logger.warning(f"Grounding check failed: {error_msg}")
                return False, error_msg

    logger.debug(
        f"Grounding check passed: {len(retrieved_chunks)} chunks retrieved")
    return True, None


def create_refusal_response(reason: str, query: str) -> str:
    """
    Create a polite refusal response based on the reason.

    Args:
        reason: Category of refusal (legal, medical, etc.)
        query: Original query

    Returns:
        Refusal message
    """
    refusal_templates = {
        "legal": (
            "I cannot provide legal advice. For legal questions, please consult "
            "with a qualified attorney. I can help answer questions about company "
            "policies and procedures based on our internal documents."
        ),
        "medical": (
            "I cannot provide medical advice or diagnoses. For medical questions, "
            "please consult with a qualified healthcare provider. I can help answer "
            "questions about company policies and procedures."
        ),
        "financial": (
            "I cannot provide financial or investment advice. For financial questions, "
            "please consult with a qualified financial advisor. I can help answer "
            "questions about company policies and procedures."
        ),
        "document_generation": (
            "I cannot generate legal documents, contracts, or letters. For document "
            "generation needs, please consult with appropriate legal or administrative "
            "resources. I can help answer questions about company policies and procedures."
        ),
    }

    template = refusal_templates.get(
        reason, "I cannot assist with this request.")
    logger.info(f"Created refusal response for category: {reason}")
    return template


def apply_guards(
    query: str,
    retrieved_chunks: Optional[List[Dict]] = None,
    file_paths: Optional[List[str]] = None
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Apply all guards to a query.

    Args:
        query: User query
        retrieved_chunks: Retrieved chunks for grounding check
        grounding_required: Whether grounding is required
        file_paths: File paths to validate (if any)

    Returns:
        (passed, refusal_message, masked_query) tuple
        - passed: True if all guards pass
        - refusal_message: Error message if guards fail
        - masked_query: Query with PII masked
    """
    logger.debug(f"Applying guards to query: '{query[:100]}...'")

    # Check refuse patterns
    should_refuse, reason = check_refuse_patterns(query)
    if should_refuse:
        refusal_msg = create_refusal_response(reason, query)
        logger.warning(f"Guard check failed: query refused - {reason}")
        return False, refusal_msg, None

    # Mask PII in query
    masked_query = mask_pii(query)

    # Check grounding if required
    if retrieved_chunks is not None:
        is_grounded, grounding_error = check_grounding_required(
            query, retrieved_chunks
        )
        if not is_grounded:
            logger.warning(f"Guard check failed: {grounding_error}")
            return False, grounding_error, masked_query

    # Validate file paths if provided
    if file_paths:
        for file_path in file_paths:
            is_valid, error = validate_file_path(file_path)
            if not is_valid:
                logger.warning(f"Guard check failed: {error}")
                return False, error, masked_query

    logger.debug("All guard checks passed")
    return True, None, masked_query


# Ensure sandbox directory exists
SANDBOX_DIR.mkdir(parents=True, exist_ok=True)
logger.info(f"Sandbox directory initialized: {SANDBOX_DIR}")
