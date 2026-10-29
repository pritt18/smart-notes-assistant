"""SQLite database management for persistent note storage.

Ensures idempotent loading of starter notes (N001-N012) and generates
distinct IDs (e.g. U001, U002, ...) for user-created notes to prevent ID collisions.
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.config import DB_PATH, STARTER_NOTES_JSON


class Database:
    """Thread-safe SQLite database manager for notes persistence."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = Path(db_path)
        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Create a sqlite connection with row dict access."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        """Initialize SQLite database tables and indexes."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    category TEXT NOT NULL,
                    uncertain INTEGER NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
            """)
            # Initialize user note sequence counter if absent
            cursor.execute("""
                INSERT OR IGNORE INTO meta (key, value) VALUES ('user_seq', '0')
            """)
            conn.commit()

    def get_next_user_id(self) -> str:
        """Generate a monotonically increasing user note ID (e.g. U001, U002).

        User-created notes will never collide with or reuse starter IDs (N001-N012).
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT value FROM meta WHERE key = 'user_seq'")
            row = cursor.fetchone()
            curr = int(row['value']) if row else 0
            nxt = curr + 1
            cursor.execute("UPDATE meta SET value = ? WHERE key = 'user_seq'", (str(nxt),))
            conn.commit()
            return f"U{nxt:03d}"

    def add_note(
        self,
        text: str,
        category: str,
        uncertain: bool,
        confidence: float,
        note_id: Optional[str] = None
    ) -> Dict:
        """Add and save a note to SQLite.

        If note_id is not specified, an auto-incremented user ID (U###) is generated.
        """
        if not text or not text.strip():
            raise ValueError("Note text cannot be empty or whitespace only.")

        clean_text = text.strip()
        assigned_id = note_id if note_id else self.get_next_user_id()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO notes (id, text, category, uncertain, confidence)
                VALUES (?, ?, ?, ?, ?)
            """, (assigned_id, clean_text, category, int(uncertain), float(confidence)))
            conn.commit()

        return {
            "id": assigned_id,
            "text": clean_text,
            "category": category,
            "uncertain": bool(uncertain),
            "confidence": round(float(confidence), 4)
        }

    def get_all_notes(self, category: Optional[str] = None, uncertain_only: bool = False) -> List[Dict]:
        """Fetch all saved notes, optionally filtered by category or uncertainty flag."""
        query = "SELECT id, text, category, uncertain, confidence, created_at FROM notes"
        params = []
        conditions = []

        if category and category.lower() != "all":
            conditions.append("category = ?")
            params.append(category)

        if uncertain_only:
            conditions.append("uncertain = 1")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY created_at DESC, id DESC"

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [
                {
                    "id": r["id"],
                    "text": r["text"],
                    "category": r["category"],
                    "uncertain": bool(r["uncertain"]),
                    "confidence": round(float(r["confidence"]), 4),
                    "created_at": r["created_at"]
                }
                for r in rows
            ]

    def get_note(self, note_id: str) -> Optional[Dict]:
        """Fetch a single note by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, text, category, uncertain, confidence, created_at FROM notes WHERE id = ?", (note_id,))
            r = cursor.fetchone()
            if not r:
                return None
            return {
                "id": r["id"],
                "text": r["text"],
                "category": r["category"],
                "uncertain": bool(r["uncertain"]),
                "confidence": round(float(r["confidence"]), 4),
                "created_at": r["created_at"]
            }

    def delete_note(self, note_id: str) -> bool:
        """Delete a note by ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM notes WHERE id = ?", (note_id,))
            conn.commit()
            return cursor.rowcount > 0

    def load_starter_notes(self, classifier) -> Tuple[int, int]:
        """Idempotently load notes from starter_notes.json using their original IDs (N001-N012).

        Returns:
            Tuple[int, int]: (inserted_count, skipped_count)
        """
        if not STARTER_NOTES_JSON.exists():
            raise FileNotFoundError(f"Starter notes file not found at {STARTER_NOTES_JSON}")

        with open(STARTER_NOTES_JSON, 'r', encoding='utf-8') as f:
            starter_data = json.load(f)

        inserted = 0
        skipped = 0

        with self._get_connection() as conn:
            cursor = conn.cursor()
            for item in starter_data:
                note_id = item["id"]
                text = item["text"]

                # Check if this ID already exists
                cursor.execute("SELECT id FROM notes WHERE id = ?", (note_id,))
                if cursor.fetchone():
                    skipped += 1
                    continue

                # Run note classification with the trained model
                pred = classifier.predict(text)
                cursor.execute("""
                    INSERT INTO notes (id, text, category, uncertain, confidence)
                    VALUES (?, ?, ?, ?, ?)
                """, (note_id, text, pred["category"], int(pred["uncertain"]), float(pred["confidence"])))
                inserted += 1

            conn.commit()

        return inserted, skipped
