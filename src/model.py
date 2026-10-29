"""Note classification model training, loading, and inference pipeline.
"""
from pathlib import Path
import csv
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import MODEL_PATH, CATEGORIES, UNCERTAINTY_THRESHOLD


def build_pipeline(c_value: float = 3.0) -> Pipeline:
    """Build a TF-IDF + Logistic Regression text classification pipeline."""
    return Pipeline([
        ('vec', TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1
        )),
        ('clf', LogisticRegression(
            C=c_value,
            max_iter=1000,
            random_state=42
        ))
    ])


def load_dataset(csv_path: Path):
    """Load text and category columns from a CSV file."""
    texts, labels = [], []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row['text'])
            if 'category' in row and row['category']:
                labels.append(row['category'])
    return texts, labels


def train_model(train_csv: Path, output_path: Path = MODEL_PATH) -> Pipeline:
    """Train the classifier pipeline and serialize it to disk."""
    texts, labels = load_dataset(train_csv)
    pipeline = build_pipeline(c_value=3.0)
    pipeline.fit(texts, labels)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)
    return pipeline


class NoteClassifier:
    """Production wrapper for note classification and uncertainty estimation."""

    def __init__(self, model_path: Path = MODEL_PATH, threshold: float = UNCERTAINTY_THRESHOLD):
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.pipeline: Pipeline = None
        self.classes: list[str] = []
        self._load()

    def _load(self):
        """Load the trained model pipeline from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. "
                "Run `python train.py` first to train and serialize the model."
            )
        self.pipeline = joblib.load(self.model_path)
        self.classes = list(self.pipeline.classes_)

    def predict(self, text: str) -> dict:
        """Predict the category and uncertainty flag for a given note text.

        Returns:
            dict containing:
                - category (str): The predicted category ('Study', 'Money', 'Travel', 'Shopping')
                - uncertain (bool): True if the model's confidence is below the uncertainty threshold
                - confidence (float): The maximum predicted class probability
                - probabilities (dict): Mapping of each class to its predicted probability
        """
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Text must be a non-empty string.")

        probabilities = self.pipeline.predict_proba([text])[0]
        max_idx = int(np.argmax(probabilities))
        best_category = str(self.classes[max_idx])
        max_confidence = float(probabilities[max_idx])

        # A note is marked uncertain if the model's top probability is below the threshold
        is_uncertain = bool(max_confidence < self.threshold)

        prob_map = {
            cls_name: round(float(prob), 4)
            for cls_name, prob in zip(self.classes, probabilities)
        }

        return {
            "category": best_category,
            "uncertain": is_uncertain,
            "confidence": round(max_confidence, 4),
            "probabilities": prob_map
        }
