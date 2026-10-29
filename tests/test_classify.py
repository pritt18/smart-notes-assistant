"""Tests for POST /classify API contract compliance and input validation.
"""
import pytest
from pydantic import ValidationError
from app.main import classify_note, ClassifyRequest, ClassifyResponse
from src.config import CATEGORIES

ALLOWED_CATEGORIES = set(CATEGORIES)


def test_classify_valid_notes():
    """Ensure normal notes return valid category and boolean uncertain flag."""
    test_cases = [
        "Revise statistics chapters before the test on Wednesday.",
        "Check my bank statement on Tuesday.",
        "Book a train to Delhi for Monday morning.",
        "Purchase a laptop stand from the store.",
        "Move ₹2,000 into my savings account."
    ]

    for note in test_cases:
        req = ClassifyRequest(text=note)
        res = classify_note(req)
        assert isinstance(res, ClassifyResponse)
        assert res.category in ALLOWED_CATEGORIES
        assert isinstance(res.uncertain, bool)


def test_classify_out_of_scope_note():
    """Ensure out-of-scope notes are flagged as uncertain."""
    req = ClassifyRequest(text="Call grandma this weekend.")
    res = classify_note(req)
    assert res.category in ALLOWED_CATEGORIES
    assert res.uncertain is True


def test_classify_empty_and_whitespace():
    """Ensure empty and whitespace-only text raise ValidationError (HTTP 422)."""
    with pytest.raises(ValidationError):
        ClassifyRequest(text="")

    with pytest.raises(ValidationError):
        ClassifyRequest(text="     \t \n  ")


def test_classify_invalid_body_types():
    """Ensure non-string and missing field bodies raise ValidationError."""
    with pytest.raises(ValidationError):
        ClassifyRequest()  # missing text field

    with pytest.raises(ValidationError):
        ClassifyRequest(text=42)  # text is a number

    with pytest.raises(ValidationError):
        ClassifyRequest(text=None)  # text is null


def test_classify_long_note():
    """Ensure a 1000-character note is processed properly."""
    long_note = "Pay the electricity bill on time. " * 30
    req = ClassifyRequest(text=long_note)
    res = classify_note(req)
    assert res.category in ALLOWED_CATEGORIES
    assert isinstance(res.uncertain, bool)


def test_classify_repeatability():
    """Ensure identical text yields identical predictions."""
    note = "Flight to Pune leaves at 6 PM tomorrow."
    res1 = classify_note(ClassifyRequest(text=note))
    res2 = classify_note(ClassifyRequest(text=note))
    assert res1.category == res2.category
    assert res1.uncertain == res2.uncertain
    assert res1.confidence == res2.confidence
