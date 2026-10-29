"""Configuration and project paths for Smart Notes Assistant.
"""
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "notes.db"

# Data file paths
TRAIN_CSV = DATA_DIR / "train.csv"
VALIDATION_CSV = DATA_DIR / "validation.csv"
OUT_OF_SCOPE_CSV = DATA_DIR / "out_of_scope_examples.csv"
STARTER_NOTES_JSON = DATA_DIR / "starter_notes.json"
SEARCH_QUERIES_JSON = DATA_DIR / "search_queries.json"

# Saved model artifact
MODEL_PATH = MODELS_DIR / "classifier.joblib"

# Categories defined by the system
CATEGORIES = ["Study", "Money", "Travel", "Shopping"]

# Uncertainty threshold for maximum predicted class probability
# Probabilities below this threshold are flagged as uncertain
UNCERTAINTY_THRESHOLD = 0.45

# Search minimum relevance score threshold
SEARCH_SCORE_THRESHOLD = 0.12
