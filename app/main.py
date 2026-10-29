"""FastAPI application for Smart Notes Assistant.

Serves the mandatory POST /classify endpoint, note management REST APIs,
semantic search, and a modern single-page web user interface.
"""
from pathlib import Path
from typing import Optional, List, Dict
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, field_validator

from src.config import CATEGORIES
from src.model import NoteClassifier
from src.db import Database
from src.search import TfidfHybridSearchEngine

# Global instances initialized lazily/reliably
_classifier: Optional[NoteClassifier] = None
_db: Optional[Database] = None
_search_engine: Optional[TfidfHybridSearchEngine] = None


def get_classifier() -> NoteClassifier:
    global _classifier
    if _classifier is None:
        _classifier = NoteClassifier()
    return _classifier


def get_db() -> Database:
    global _db
    if _db is None:
        _db = Database()
    return _db


def get_search_engine() -> TfidfHybridSearchEngine:
    global _search_engine
    if _search_engine is None:
        _search_engine = TfidfHybridSearchEngine()
    return _search_engine


app = FastAPI(
    title="Smart Notes Assistant",
    description="Intelligent note categorization, persistence, and semantic retrieval service.",
    version="1.0.0"
)

# Mount static files
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# Pydantic schemas
class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Note text up to 1000 characters.")

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Text must be a non-empty string containing non-whitespace characters.")
        if len(v) > 2000:
            raise ValueError("Text exceeds maximum allowed length.")
        return v


class ClassifyResponse(BaseModel):
    category: str
    uncertain: bool
    confidence: Optional[float] = None
    probabilities: Optional[Dict[str, float]] = None


class CreateNoteRequest(BaseModel):
    text: str

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Note text cannot be empty or whitespace only.")
        return v.strip()


# Exception handler for clean 4xx responses without non-serializable objects
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    formatted_errors = [
        {"loc": list(err.get("loc", [])), "msg": str(err.get("msg", "Invalid value"))}
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": formatted_errors, "message": "Invalid note input."}
    )


# -----------------------------------------------------------------------------
# 1. API Contract Endpoint: POST /classify
# -----------------------------------------------------------------------------
@app.post("/classify", response_model=ClassifyResponse)
def classify_note(req: ClassifyRequest):
    """Classify note text into Study, Money, Travel, or Shopping.

    Returns category and uncertain flag according to API_CONTRACT.md.
    """
    clf = get_classifier()
    try:
        prediction = clf.predict(req.text)
        return ClassifyResponse(
            category=prediction["category"],
            uncertain=prediction["uncertain"],
            confidence=prediction["confidence"],
            probabilities=prediction["probabilities"]
        )
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


# -----------------------------------------------------------------------------
# 2. Note Management REST APIs
# -----------------------------------------------------------------------------
@app.get("/api/notes")
def list_notes(
    category: Optional[str] = Query(None, description="Filter by category (Study, Money, Travel, Shopping, All)"),
    uncertain_only: bool = Query(False, description="Filter for uncertain notes only")
):
    """Retrieve all saved notes with optional filtering."""
    db_inst = get_db()
    notes = db_inst.get_all_notes(category=category, uncertain_only=uncertain_only)
    return {"notes": notes, "total": len(notes)}


@app.post("/api/notes", status_code=status.HTTP_201_CREATED)
def create_note(req: CreateNoteRequest):
    """Add a new note, classify it, flag uncertainty, and persist to SQLite."""
    clf = get_classifier()
    db_inst = get_db()

    prediction = clf.predict(req.text)
    saved = db_inst.add_note(
        text=req.text,
        category=prediction["category"],
        uncertain=prediction["uncertain"],
        confidence=prediction["confidence"]
    )
    return saved


@app.get("/api/notes/{note_id}")
def get_note(note_id: str):
    """Retrieve a single note by ID."""
    db_inst = get_db()
    note = db_inst.get_note(note_id)
    if not note:
        raise HTTPException(status_code=404, detail="Note not found.")
    return note


@app.delete("/api/notes/{note_id}")
def delete_note(note_id: str):
    """Delete a note by ID."""
    db_inst = get_db()
    deleted = db_inst.delete_note(note_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Note not found.")
    return {"status": "success", "deleted_id": note_id}


@app.post("/api/notes/load-starter")
def load_starter_notes():
    """Load sample notes from data/starter_notes.json idempotently."""
    clf = get_classifier()
    db_inst = get_db()
    inserted, skipped = db_inst.load_starter_notes(clf)
    return {
        "status": "success",
        "inserted": inserted,
        "skipped": skipped,
        "message": f"Loaded {inserted} new starter notes ({skipped} already present)."
    }


# -----------------------------------------------------------------------------
# 3. Search API
# -----------------------------------------------------------------------------
@app.get("/api/search")
def search_notes(q: str = Query(..., min_length=1, description="Search query")):
    """Search notes using TF-IDF hybrid vector similarity and relevance filtering."""
    db_inst = get_db()
    engine = get_search_engine()

    all_notes = db_inst.get_all_notes()
    results = engine.search(query=q, notes=all_notes, top_k=5)
    return {
        "query": q,
        "results": results,
        "count": len(results),
        "message": "Results found." if results else "No matching notes found."
    }


# -----------------------------------------------------------------------------
# 4. Web UI Route
# -----------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the single-page application frontend."""
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return HTMLResponse("<h3>Smart Notes Assistant API is running. UI index.html not found.</h3>")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))
