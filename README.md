# Smart Notes Assistant

An intelligent, lightweight note-taking application powered by machine learning text categorization, semantic vector retrieval, and uncertainty estimation. Built with Python 3.11, FastAPI, scikit-learn, and SQLite.

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)
![Tests](https://img.shields.io/badge/Tests-14%20Passed-brightgreen.svg)

---

## Key Features

- **Automated Note Categorization:** Automatically predicts note categories (`Study`, `Money`, `Travel`, `Shopping`) using a trained TF-IDF + Logistic Regression pipeline.
- **Uncertainty & Out-of-Scope Detection:** Calibrated probability thresholding ($\tau = 0.45$) identifies ambiguous notes or out-of-scope entries (e.g. *"Call grandma this weekend"*) and marks them with an explicit `⚠️ Uncertain` badge rather than forcing an inaccurate category.
- **Semantic & Hybrid Search:** Combines word-level TF-IDF, subword character n-grams (3–5 chars), and domain semantic query expansion to achieve **100% Hit Rate @3** on benchmark queries, including queries sharing zero content words with their target notes (e.g. *"money transfer"* retrieving *"Move ₹2,000 into savings"*).
- **Persistent Storage:** SQLite database with connection management, restart durability, idempotent starter notes loading (`N001`–`N012`), and guaranteed collision-free user note IDs (`U001`, `U002`...).
- **Standardized API Contract:** Fully compliant `POST /classify` endpoint operating with sub-30ms latency, strict 4xx validation handling, and Unicode support (`₹`, etc.).
- **Responsive Web Interface:** Modern, clean single-page application served directly from the FastAPI server without external CDN or cloud dependencies.

---

## Architecture Overview

```
smart-notes-assistant/
│
├── data/
│   ├── train.csv                 # 156 labeled training examples
│   ├── validation.csv            # 50 labeled validation examples
│   ├── out_of_scope_examples.csv # 12 out-of-scope examples for threshold tuning
│   ├── starter_notes.json        # 12 sample starter notes (N001-N012)
│   └── search_queries.json       # 12 benchmark search queries (4 zero-overlap)
│
├── models/
│   └── classifier.joblib         # Serialized production pipeline artifact
│
├── src/                          # Core backend logic
│   ├── config.py                 # Hyperparameters, constants, thresholds
│   ├── model.py                  # Training pipeline, inference, uncertainty logic
│   ├── db.py                     # SQLite persistence, ID generation, starter loader
│   └── search.py                 # Keyword baseline & Hybrid TF-IDF search engine
│
├── app/                          # Web application & REST endpoints
│   ├── main.py                   # FastAPI service: /classify, /api/notes, /api/search
│   └── static/                   # Frontend SPA (HTML, CSS, Vanilla JS)
│       ├── index.html
│       ├── style.css
│       └── app.js
│
├── tests/                        # Comprehensive test suite
│   ├── test_classify.py          # /classify contract, 4xx validation, edge cases
│   ├── test_storage.py           # Persistence, restart durability, ID safety
│   └── test_search.py            # Search ranking, relevance thresholds, zero-match
│
├── examples/
│   └── test_api.py               # Official API contract smoke test
│
├── train.py                      # Model training & validation reproducibility script
├── evaluate_search.py            # Search Hit Rate @3 benchmark script
├── load_starter_notes.py         # CLI tool to load starter notes idempotently
├── run.py                        # Server startup entry point
├── requirements.txt              # Pinned Python package dependencies
├── RESULTS.md                    # Detailed empirical evaluation results & error analysis
└── AI_WORKFLOW.md                # Development process & tool review log
```

---

## Quickstart

### 1. Prerequisites & Environment Setup

Python 3.10+ is recommended. Clone the repository and install dependencies:

```bash
# Clone the repository
git clone https://github.com/pritt18/smart-notes-assistant.git
cd smart-notes-assistant

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

---

## Reproducing Model Training & Evaluation

To train the classification model from `data/train.csv`, evaluate baseline vs trained models on `data/validation.csv`, print detailed error analyses with empirical feature evidence, and serialize `models/classifier.joblib`, run:

```bash
python train.py
```

### Summary of Model Performance

| Approach | Accuracy | Macro-F1 | Notes |
| :--- | :---: | :---: | :--- |
| **Baseline (Majority Class)** | 0.3000 | 0.1154 | Naive most frequent class (`Study`) |
| **Initial Model (Unigram, $C=1.0$)** | 0.6800 | 0.7079 | Suffers from stopwords & temporal bias |
| **Production Model (TF-IDF + LogReg $C=3.0$)** | **0.8800** | **0.8804** | Stop words removed, bigrams, sublinear TF |
| **Alternative Model (Complement Naive Bayes)** | **0.9200** | **0.9158** | High accuracy on sparse vocabularies |

Full per-class precision/recall, confusion matrix, error breakdowns, and mathematical evidence are documented in [RESULTS.md](RESULTS.md).

---

## Reproducing Search Quality Benchmark

To run the Hit Rate @3 benchmark across the 12 starter notes using both the Keyword Baseline and TF-IDF Hybrid Vector Search:

```bash
python evaluate_search.py
```

### Search Benchmark Results

| Approach | Hit Rate @3 | Notes |
| :--- | :---: | :--- |
| **Keyword Baseline (Token Overlap)** | 66.7% (8 / 12) | Fails on all 4 zero-content-word overlap queries |
| **TF-IDF Hybrid (Subword + Semantic Expansion)** | **100.0% (12 / 12)** | Retrieves all synonym and semantic target notes |

---

## Starting the Application

Start the web application and API server with a single command:

```bash
python run.py --host 127.0.0.1 --port 8000
```

Once started:
- **Interactive Web UI:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive API Documentation (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Classification Endpoint:** `POST http://127.0.0.1:8000/classify`

---

## Loading Starter Notes

You can load the 12 sample notes (`N001`–`N012`) into persistent storage in two ways:

1. **Via Web Interface:** Click the **"Load Starter Notes"** button in the top navigation bar.
2. **Via Command Line:**
   ```bash
   python load_starter_notes.py
   ```

*Note: Loading starter notes is completely idempotent—running it multiple times will not duplicate existing entries. User-created notes receive a `U###` prefix and will never collide with `N001`–`N012`.*

---

## Testing & Verification

### Running Automated Test Suite

Execute the unit and integration tests covering API contracts, input validation, persistence, and search ranking:

```bash
pytest -v
```

Expected output:
```
============================= 14 passed in 2.74s ==============================
```

### Running API Smoke Test

With the server running on port 8000, verify the API contract using the smoke test script:

```bash
python examples/test_api.py --url http://127.0.0.1:8000/classify
```

Output checks:
```
PASS valid note 'Revise for my biology examination.' -> HTTP 200 (6 ms)
PASS valid note 'Check my bank balance.' -> HTTP 200 (5 ms)
PASS valid note 'Book a train to Pune.' -> HTTP 200 (19 ms)
PASS valid note 'Purchase a pair of shoes.' -> HTTP 200 (6 ms)
PASS valid note 'Move ₹2,000 into savings before the 1st.' -> HTTP 200 (27 ms)
PASS valid note 'Call grandma this weekend.' -> HTTP 200, uncertain: True (30 ms)
PASS whitespace text -> HTTP 422 (expected 4xx)
PASS empty text -> HTTP 422 (expected 4xx)
PASS missing text field -> HTTP 422 (expected 4xx)
PASS text is a number -> HTTP 422 (expected 4xx)
PASS text is null -> HTTP 422 (expected 4xx)
PASS malformed JSON -> HTTP 422 (expected 4xx)
PASS server still answers after invalid requests
PASS same note gives the same response twice

All format checks passed.
```

---

## API Contract Reference

### POST `/classify`

Classifies note text into one of the four categories and indicates whether the model is uncertain.

#### Request Body
```json
{
  "text": "Revise Python loops before Friday."
}
```

#### Successful Response (`HTTP 200 OK`)
```json
{
  "category": "Study",
  "uncertain": false,
  "confidence": 0.8924,
  "probabilities": {
    "Study": 0.8924,
    "Money": 0.0341,
    "Travel": 0.0412,
    "Shopping": 0.0323
  }
}
```

#### Out-of-Scope Response (`HTTP 200 OK`)
```json
{
  "text": "Call grandma this weekend."
}
```
Response:
```json
{
  "category": "Study",
  "uncertain": true,
  "confidence": 0.3106,
  "probabilities": { ... }
}
```

#### Invalid Input (`HTTP 422 Unprocessable Entity` / `HTTP 400 Bad Request`)
If `text` is empty, whitespace only, missing, or not a string, a clean 4xx response is returned without server crash.

---

## Additional REST Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Web UI single-page application |
| `GET` | `/api/notes` | List saved notes (supports `?category=Study` and `?uncertain_only=true`) |
| `POST` | `/api/notes` | Create, classify, and persist a new note |
| `GET` | `/api/notes/{id}` | Retrieve a specific note by ID |
| `DELETE` | `/api/notes/{id}` | Delete a note by ID |
| `POST` | `/api/notes/load-starter` | Idempotently load starter notes (`N001`–`N012`) |
| `GET` | `/api/search?q={query}` | Search notes using Hybrid TF-IDF vector retrieval |

---

## Documentation Links

- [RESULTS.md](RESULTS.md): Empirical metrics, error analysis with feature evidence, search benchmarks, and out-of-scope calibration.
- [AI_WORKFLOW.md](AI_WORKFLOW.md): Development methodology, AI-assisted tooling review, and bug investigation report.
