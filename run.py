#!/usr/bin/env python3
"""Main entry point to start the Smart Notes Assistant server.

Usage:
    python run.py [--port 8000] [--host 127.0.0.1]
"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import argparse
import uvicorn
from pathlib import Path
from src.config import MODEL_PATH, TRAIN_CSV
from src.model import train_model


def ensure_model_exists():
    """Ensure that the classification model is trained and available."""
    if not MODEL_PATH.exists():
        print(f"Model artifact not found at {MODEL_PATH}. Training initial model from {TRAIN_CSV}...")
        train_model(TRAIN_CSV, MODEL_PATH)
        print("[OK] Model training complete.")


def main():
    parser = argparse.ArgumentParser(description="Smart Notes Assistant Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload on code change")
    args = parser.parse_args()

    ensure_model_exists()

    print("\n" + "=" * 65)
    print("  Smart Notes Assistant is starting up!")
    print(f"  Web Application:  http://{args.host}:{args.port}")
    print(f"  API Docs:         http://{args.host}:{args.port}/docs")
    print(f"  API Endpoint:     http://{args.host}:{args.port}/classify")
    print("=" * 65 + "\n")

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == '__main__':
    main()
