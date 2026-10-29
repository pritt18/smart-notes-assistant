"""Pytest fixtures and configuration.
"""
import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODEL_PATH, TRAIN_CSV
from src.model import train_model
from src.db import Database


@pytest.fixture(scope="session", autouse=True)
def setup_model():
    """Ensure classification model exists for tests."""
    if not MODEL_PATH.exists():
        train_model(TRAIN_CSV, MODEL_PATH)


@pytest.fixture
def temp_db(tmp_path):
    """Temporary SQLite database fixture for isolated tests."""
    test_db_file = tmp_path / "test_notes.db"
    return Database(db_path=test_db_file)
