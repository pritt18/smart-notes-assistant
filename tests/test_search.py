"""Tests for search quality, relevance scoring, and zero-match filtering.
"""
import pytest
from src.search import KeywordSearchBaseline, TfidfHybridSearchEngine

SAMPLE_NOTES = [
    {"id": "N001", "text": "My Python test is on Friday at 10 AM.", "category": "Study", "uncertain": False},
    {"id": "N002", "text": "Pay the electricity bill by the 28th.", "category": "Money", "uncertain": False},
    {"id": "N003", "text": "Our flight to Delhi is on Monday morning.", "category": "Travel", "uncertain": False},
    {"id": "N006", "text": "Move ₹2,000 into the savings account this month.", "category": "Money", "uncertain": False},
    {"id": "N007", "text": "The hotel in Jaipur is booked for Saturday.", "category": "Travel", "uncertain": False},
    {"id": "N012", "text": "Buy rice, eggs, and vegetables from the supermarket.", "category": "Shopping", "uncertain": False},
]


def test_keyword_baseline_exact_match():
    """Keyword baseline matches notes sharing tokens."""
    engine = KeywordSearchBaseline()
    results = engine.search("flight to Delhi", SAMPLE_NOTES)
    assert len(results) > 0
    assert results[0]["id"] == "N003"


def test_hybrid_search_semantic_expansion():
    """Hybrid search uses domain semantic expansion to match zero-word-overlap queries."""
    engine = TfidfHybridSearchEngine(min_score=0.05)

    # "money transfer" shares 0 words with N006 ("Move ₹2,000 into the savings account this month.")
    results = engine.search("money transfer", SAMPLE_NOTES, top_k=3)
    assert len(results) > 0
    top_ids = [r["id"] for r in results]
    assert "N006" in top_ids


def test_search_filters_irrelevant_queries():
    """Out-of-domain queries with no relevance score should return empty list."""
    engine = TfidfHybridSearchEngine(min_score=0.15)
    results = engine.search("scuba diving coral reef submarine exploration", SAMPLE_NOTES)
    assert len(results) == 0, "Expected low-relevance queries to return no results."


def test_empty_query_returns_empty():
    """Empty or whitespace queries return empty list."""
    engine = TfidfHybridSearchEngine()
    assert engine.search("", SAMPLE_NOTES) == []
    assert engine.search("    ", SAMPLE_NOTES) == []
