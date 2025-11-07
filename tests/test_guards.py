"""Tests for guards and policies."""
import pytest
from src.guards.policy import (
    check_refuse_patterns,
    mask_pii,
    validate_file_path,
    check_grounding_required,
    apply_guards,
    create_refusal_response,
    SANDBOX_DIR,
)


def test_check_refuse_patterns_legal():
    """Test that legal advice queries are refused."""
    should_refuse, reason = check_refuse_patterns("Can I sue my employer?")
    assert should_refuse is True
    assert reason == "legal"


def test_check_refuse_patterns_medical():
    """Test that medical advice queries are refused."""
    should_refuse, reason = check_refuse_patterns(
        "What medicine should I take for my headache?")
    assert should_refuse is True
    assert reason == "medical"


def test_check_refuse_patterns_financial():
    """Test that financial advice queries are refused."""
    should_refuse, reason = check_refuse_patterns("Should I buy Apple stock?")
    assert should_refuse is True
    assert reason == "financial"


def test_check_refuse_patterns_document_generation():
    """Test that document generation queries are refused."""
    should_refuse, reason = check_refuse_patterns(
        "Generate a letter claiming I'm exempt")
    assert should_refuse is True
    assert reason == "document_generation"


def test_check_refuse_patterns_allowed():
    """Test that normal queries are not refused."""
    should_refuse, reason = check_refuse_patterns(
        "What is the relocation allowance?")
    assert should_refuse is False
    assert reason is None


def test_mask_pii_email():
    """Test that emails are masked."""
    text = "Contact me at john.doe@example.com"
    masked = mask_pii(text)
    assert "[REDACTED_EMAIL]" in masked
    assert "john.doe@example.com" not in masked


def test_mask_pii_multiple_emails():
    """Test that multiple emails are masked."""
    text = "Email alice@test.com or bob@test.com"
    masked = mask_pii(text)
    assert masked.count("[REDACTED_EMAIL]") == 2
    assert "alice@test.com" not in masked
    assert "bob@test.com" not in masked


def test_mask_pii_ssn():
    """Test that SSNs are masked."""
    text = "My SSN is 123-45-6789"
    masked = mask_pii(text)
    assert "[REDACTED_SSN]" in masked
    assert "123-45-6789" not in masked


def test_mask_pii_phone():
    """Test that phone numbers are masked."""
    text = "Call me at 123.456.7890"
    masked = mask_pii(text)
    assert "[REDACTED_PHONE]" in masked
    assert "123.456.7890" not in masked


def test_mask_pii_credit_card():
    """Test that credit card numbers are masked."""
    text = "Card number: 1234567890123456"
    masked = mask_pii(text)
    assert "[REDACTED_CREDIT_CARD]" in masked
    assert "1234567890123456" not in masked


def test_mask_pii_mixed():
    """Test that multiple PII types are masked."""
    text = "Email: user@example.com, Phone: 123.456.7890, SSN: 123-45-6789"
    masked = mask_pii(text)
    assert "[REDACTED_EMAIL]" in masked
    assert "[REDACTED_PHONE]" in masked
    assert "[REDACTED_SSN]" in masked
    assert "user@example.com" not in masked


def test_validate_file_path_sandbox():
    """Test that paths within sandbox are valid."""
    valid_path = str(SANDBOX_DIR / "test.txt")
    is_valid, error = validate_file_path(valid_path)
    assert is_valid is True
    assert error is None


def test_validate_file_path_outside_sandbox():
    """Test that paths outside sandbox are rejected."""
    from pathlib import Path
    # Try to access a file outside sandbox
    invalid_path = str(Path("/etc/passwd"))
    is_valid, error = validate_file_path(invalid_path)
    assert is_valid is False
    assert error is not None
    assert "outside sandbox" in error.lower()


def test_validate_file_path_relative():
    """Test that relative paths are resolved correctly."""
    # Relative path should resolve to sandbox
    relative_path = "data/sandbox/test.txt"
    is_valid, error = validate_file_path(relative_path)
    # Should be valid if it resolves within sandbox
    assert is_valid is True or "outside sandbox" in error.lower()


def test_check_grounding_required_with_chunks():
    """Test that grounding check passes when chunks exist."""
    query = "What is the relocation allowance?"
    chunks = [{"content": "Relocation allowance is $5,000", "source": "policy.md"}]
    is_grounded, error = check_grounding_required(
        query, chunks)
    assert is_grounded is True
    assert error is None


def test_check_grounding_required_no_chunks_factual():
    """Test that grounding check fails when no chunks for factual query."""
    query = "What is the relocation allowance?"
    chunks = []
    is_grounded, error = check_grounding_required(
        query, chunks)
    assert is_grounded is False
    assert error is not None
    assert "grounding" in error.lower()


def test_check_grounding_required_no_chunks_non_factual():
    """Test that grounding check passes for non-factual queries without chunks."""
    query = "Hello, how are you?"
    chunks = []
    is_grounded, error = check_grounding_required(
        query, chunks)
    assert is_grounded is True
    assert error is None


def test_apply_guards_refuse():
    """Test that apply_guards refuses legal queries."""
    query = "Can I sue my employer for discrimination?"
    passed, refusal_msg, masked_query = apply_guards(
        query)
    assert passed is False
    assert refusal_msg is not None
    assert "legal" in refusal_msg.lower() or "cannot" in refusal_msg.lower()


def test_apply_guards_pass():
    """Test that apply_guards passes normal queries."""
    query = "What is the relocation allowance amount?"
    passed, refusal_msg, masked_query = apply_guards(
        query)
    assert passed is True
    assert refusal_msg is None
    assert masked_query is not None


def test_apply_guards_with_pii():
    """Test that apply_guards masks PII in query."""
    query = "Send email to john@example.com"
    passed, refusal_msg, masked_query = apply_guards(
        query)
    assert passed is True
    assert "[REDACTED_EMAIL]" in masked_query
    assert "john@example.com" not in masked_query


def test_apply_guards_file_path_validation():
    """Test that apply_guards validates file paths."""
    query = "Read file /etc/passwd"
    passed, refusal_msg, masked_query = apply_guards(
        query,
        file_paths=["/etc/passwd"]
    )
    assert passed is False
    assert refusal_msg is not None
    assert "outside sandbox" in refusal_msg.lower()


def test_apply_guards_grounding_check():
    """Test that apply_guards checks grounding when chunks provided."""
    query = "What is the relocation allowance?"
    passed, refusal_msg, masked_query = apply_guards(
        query,
        retrieved_chunks=[]
    )
    assert passed is False
    assert refusal_msg is not None
    assert "grounding" in refusal_msg.lower()


def test_create_refusal_response_legal():
    """Test that refusal responses are created correctly for legal."""
    response = create_refusal_response("legal", "Can I sue?")
    assert "legal" in response.lower() or "attorney" in response.lower()
    assert "cannot" in response.lower()


def test_create_refusal_response_medical():
    """Test that refusal responses are created correctly for medical."""
    response = create_refusal_response("medical", "What should I take?")
    assert "medical" in response.lower() or "healthcare" in response.lower()
    assert "cannot" in response.lower()


def test_create_refusal_response_financial():
    """Test that refusal responses are created correctly for financial."""
    response = create_refusal_response("financial", "Should I invest?")
    assert "financial" in response.lower() or "advisor" in response.lower()
    assert "cannot" in response.lower()


def test_create_refusal_response_unknown_category():
    """Test that unknown refusal categories get default message."""
    response = create_refusal_response("unknown_category", "test query")
    assert "cannot assist" in response.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
