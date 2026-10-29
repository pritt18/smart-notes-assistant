"""Tests for SQLite persistence, restart durability, and ID collision safety.
"""
import pytest
from src.db import Database
from src.model import NoteClassifier


def test_note_creation_and_persistence(temp_db):
    """Ensure notes can be added, saved to SQLite, and retrieved accurately."""
    saved = temp_db.add_note(
        text="Review biology exam chapters.",
        category="Study",
        uncertain=False,
        confidence=0.912
    )
    assert saved["id"] == "U001"
    assert saved["category"] == "Study"
    assert saved["uncertain"] is False

    # Retrieve from DB
    retrieved = temp_db.get_note("U001")
    assert retrieved is not None
    assert retrieved["text"] == "Review biology exam chapters."
    assert retrieved["category"] == "Study"
    assert retrieved["uncertain"] is False


def test_restart_persistence(tmp_path):
    """Simulate app restart by creating a new Database connection to same DB file."""
    db_file = tmp_path / "persistence_test.db"

    # Session 1: Create and write notes
    db1 = Database(db_path=db_file)
    db1.add_note("Pay electricity bill", "Money", False, 0.88)
    db1.add_note("Call grandma this weekend", "Study", True, 0.32)

    notes_before = db1.get_all_notes()
    assert len(notes_before) == 2

    # Session 2: Fresh instance (simulating server restart)
    db2 = Database(db_path=db_file)
    notes_after = db2.get_all_notes()
    assert len(notes_after) == 2

    # Verify fields survived restart
    note_ids = {n["id"] for n in notes_after}
    assert "U001" in note_ids
    assert "U002" in note_ids

    uncertain_notes = [n for n in notes_after if n["uncertain"]]
    assert len(uncertain_notes) == 1
    assert uncertain_notes[0]["id"] == "U002"


def test_starter_notes_idempotency_and_no_id_collision(temp_db):
    """Ensure starter notes (N001-N012) can be loaded idempotently without duplicate rows,

    and user-created notes never reuse or collide with N001-N012.
    """
    classifier = NoteClassifier()

    # First load
    inserted, skipped = temp_db.load_starter_notes(classifier)
    assert inserted == 12
    assert skipped == 0

    all_notes = temp_db.get_all_notes()
    assert len(all_notes) == 12

    # Verify original IDs are preserved (N001 - N012)
    starter_ids = {f"N{i:03d}" for i in range(1, 13)}
    stored_ids = {n["id"] for n in all_notes}
    assert starter_ids == stored_ids

    # Second load (must be idempotent: 0 inserted, 12 skipped)
    inserted2, skipped2 = temp_db.load_starter_notes(classifier)
    assert inserted2 == 0
    assert skipped2 == 12
    assert len(temp_db.get_all_notes()) == 12

    # User adds a note: verify ID is U### and never reuses N###
    user_note = temp_db.add_note("Buy groceries tomorrow", "Shopping", False, 0.95)
    assert user_note["id"].startswith("U")
    assert user_note["id"] not in starter_ids
    assert len(temp_db.get_all_notes()) == 13


def test_category_and_uncertainty_filters(temp_db):
    """Ensure category filtering and uncertainty filtering work properly."""
    temp_db.add_note("Math assignment", "Study", False, 0.95)
    temp_db.add_note("History exam", "Study", True, 0.40)
    temp_db.add_note("Pay water bill", "Money", False, 0.90)

    # Filter by Study
    study_notes = temp_db.get_all_notes(category="Study")
    assert len(study_notes) == 2

    # Filter by Money
    money_notes = temp_db.get_all_notes(category="Money")
    assert len(money_notes) == 1

    # Filter uncertain only
    uncertain_notes = temp_db.get_all_notes(uncertain_only=True)
    assert len(uncertain_notes) == 1
    assert uncertain_notes[0]["text"] == "History exam"
