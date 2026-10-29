#!/usr/bin/env python3
"""CLI utility to idempotently load starter notes (N001-N012) into the database.
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.db import Database
from src.model import NoteClassifier


def main():
    print("Loading starter notes from data/starter_notes.json...")
    classifier = NoteClassifier()
    db = Database()
    inserted, skipped = db.load_starter_notes(classifier)
    print(f"[OK] Successfully loaded {inserted} starter notes (skipped {skipped} existing).")


if __name__ == '__main__':
    main()
