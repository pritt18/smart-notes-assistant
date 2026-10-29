# AI Engineering Development Workflow

### 1. Tools
- **Environment & IDE:** VS Code with GitHub Copilot / Cursor autocomplete, Python 3.11.9, pytest, uvicorn.
- **Libraries:** FastAPI, Pydantic v2, scikit-learn, SQLite3, joblib.

---

### 2. One delegated task

#### Prompt provided
> "Generate a FastAPI route for POST /classify that accepts a JSON body with a non-empty 'text' string (up to 1000 characters), validates input, and returns category and an uncertain boolean flag according to our trained scikit-learn classifier pipeline."

#### What I kept
- The clean Pydantic schema structure (`ClassifyRequest`, `ClassifyResponse`) and standard route decorators.
- The field validation decorator checking for string instances.

#### What I changed
- Copilot suggested loading the model pipeline inside the request handler (`joblib.load("models/classifier.joblib")` on every POST request). This violated the requirement for sub-second latency and thread-safety under concurrent load. I refactored the model into a singleton `NoteClassifier` instantiated once at startup and reused across requests.
- Added strict non-whitespace validation (`v.strip()`) so whitespace-only bodies like `"   "` cleanly return 4xx errors instead of reaching the ML inference step.

---

### 3. One problem caught

#### Symptom
When testing the API using the smoke test script (`examples/test_api.py`), valid notes passed with low latency (5–30 ms), but whitespace and empty input requests failed with `HTTP 500 Internal Server Error` instead of the required 4xx response:
```
FAIL whitespace text -> HTTP 500 (expected 4xx)
FAIL empty text -> HTTP 500 (expected 4xx)
```

#### How I noticed
Inspecting the uvicorn traceback revealed:
```python
TypeError: Object of type ValueError is not JSON serializable
```
In Pydantic v2, custom `@field_validator` exceptions place the raw Python `ValueError` instance inside the `ctx` dictionary (`err['ctx']['error']`). When our custom `@app.exception_handler(RequestValidationError)` passed `exc.errors()` directly into Starlette's `JSONResponse`, standard `json.dumps` threw a serialization error, turning what should have been an HTTP 422 error into an uncaught 500 server crash.

#### How I proved the fix worked
1. Modified the exception handler in `app/main.py` to extract only serializable strings (`loc` and `msg`):
   ```python
   formatted_errors = [
       {"loc": list(err.get("loc", [])), "msg": str(err.get("msg", "Invalid value"))}
       for err in exc.errors()
   ]
   return JSONResponse(
       status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
       content={"detail": formatted_errors, "message": "Invalid note input."}
   )
   ```
2. Re-ran `examples/test_api.py` against the running server:
   ```
   PASS whitespace text -> HTTP 422 (expected 4xx)
   PASS empty text -> HTTP 422 (expected 4xx)
   All format checks passed.
   ```
3. Verified full test suite with `pytest`: all 14 tests passed in 2.74 seconds.

---

**Approximate time spent:** ~6.5 hours (model exploration, pipeline tuning, API contract implementation, search benchmarking, SQLite persistence, and responsive UI).
