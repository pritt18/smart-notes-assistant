"""Search engines for note retrieval: Keyword Overlap Baseline and TF-IDF Hybrid Vector Search.
"""
import re
from typing import List, Dict, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import SEARCH_SCORE_THRESHOLD

COMMON_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "as", "at", "be", "because", "been", "before", "being", "below",
    "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down",
    "during", "each", "few", "for", "from", "further", "had", "has", "have",
    "having", "he", "her", "here", "hers", "herself", "him", "himself", "his",
    "how", "i", "if", "in", "into", "is", "it", "its", "itself", "just", "me",
    "more", "most", "my", "myself", "no", "nor", "not", "now", "of", "off", "on",
    "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "t", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "we", "were", "what", "when", "where", "which", "while", "who",
    "whom", "why", "will", "with", "you", "your", "yours", "yourself", "yourselves"
}

# Domain semantic expansion mapping for cross-vocabulary retrieval
DOMAIN_SYNONYMS = {
    "money": ["savings", "cash", "funds", "account", "₹", "bank"],
    "transfer": ["move", "deposit", "send", "wire", "remit"],
    "grocery": ["vegetables", "rice", "eggs", "supermarket", "food", "produce"],
    "shopping": ["buy", "purchase", "order", "store", "cart"],
    "lodging": ["hotel", "room", "stay", "accommodation", "resort"],
    "reservation": ["booked", "booking", "reserved"],
    "audio": ["headphones", "earphones", "sound", "headset"],
    "gear": ["equipment", "stand", "device", "gadget"],
    "broken": ["defective", "faulty", "damaged", "return"]
}


def tokenize(text: str) -> List[str]:
    """Extract lowercase alphanumeric tokens from text."""
    return re.findall(r'[a-zA-Z0-9₹]+', text.lower())


def expand_query(query: str) -> str:
    """Expand query terms with domain synonyms to handle vocabulary mismatch."""
    tokens = tokenize(query)
    expanded = list(tokens)
    for token in tokens:
        if token in DOMAIN_SYNONYMS:
            expanded.extend(DOMAIN_SYNONYMS[token])
    return " ".join(expanded)


class KeywordSearchBaseline:
    """Baseline search algorithm based on lexical token overlap (Jaccard similarity)."""

    def search(self, query: str, notes: List[Dict], top_k: int = 3) -> List[Dict]:
        """Rank notes by exact token overlap with query tokens."""
        if not query or not query.strip() or not notes:
            return []

        q_tokens = set(tokenize(query)) - COMMON_STOP_WORDS
        if not q_tokens:
            q_tokens = set(tokenize(query))

        ranked = []
        for note in notes:
            n_tokens = set(tokenize(note['text'])) - COMMON_STOP_WORDS
            if not n_tokens:
                n_tokens = set(tokenize(note['text']))

            intersection = q_tokens & n_tokens
            union = q_tokens | n_tokens
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard > 0:
                ranked.append({
                    "id": note["id"],
                    "text": note["text"],
                    "category": note.get("category", ""),
                    "uncertain": note.get("uncertain", False),
                    "score": round(jaccard, 4)
                })

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return ranked[:top_k]


class TfidfHybridSearchEngine:
    """Production search engine combining semantic query expansion, word-level TF-IDF,

    and character n-gram matching.
    Supports typo-tolerant retrieval, morphological variations (e.g. booked/booking),
    and relevance thresholding to discard unrelated matches.
    """

    def __init__(self, min_score: float = SEARCH_SCORE_THRESHOLD):
        self.min_score = min_score
        self.word_vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        self.char_vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(3, 5),
            sublinear_tf=True
        )

    def search(self, query: str, notes: List[Dict], top_k: int = 3) -> List[Dict]:
        """Perform hybrid vector space search across candidate notes."""
        if not query or not query.strip() or not notes:
            return []

        clean_query = query.strip()
        expanded_q = expand_query(clean_query)
        texts = [n['text'] for n in notes]

        # Fit vectorizers on the active note corpus + expanded query
        corpus = texts + [expanded_q]

        try:
            w_matrix = self.word_vectorizer.fit_transform(corpus)
            c_matrix = self.char_vectorizer.fit_transform(corpus)
        except ValueError:
            return []

        q_w = w_matrix[-1:]
        notes_w = w_matrix[:-1]
        w_sims = cosine_similarity(q_w, notes_w)[0]

        q_c = c_matrix[-1:]
        notes_c = c_matrix[:-1]
        c_sims = cosine_similarity(q_c, notes_c)[0]

        # Weighted hybrid combination
        hybrid_scores = 0.65 * w_sims + 0.35 * c_sims

        results = []
        for i, score in enumerate(hybrid_scores):
            # Only return results that exceed the minimal relevance threshold
            if score >= self.min_score:
                results.append({
                    "id": notes[i]["id"],
                    "text": notes[i]["text"],
                    "category": notes[i].get("category", ""),
                    "uncertain": notes[i].get("uncertain", False),
                    "score": round(float(score), 4)
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
