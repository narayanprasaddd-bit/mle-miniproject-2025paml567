# API test collection — Deliverable 3

Satisfies the brief's requirement for *"sample request/response test calls
(e.g., Postman collection or curl commands)."*

## Run it

Terminal 1:
```bash
uvicorn src.serving.app:app --port 8000
```

Terminal 2 (Git Bash on Windows — curl ships with Git for Windows):
```bash
bash api_tests/curl_collection.sh
```

Against a different host:
```bash
BASE=http://localhost:9000 bash api_tests/curl_collection.sh
```

Exit code is 0 only if every call returned its expected status.

## What it covers — 19 calls in seven groups

| Group | Calls | Demonstrates |
|---|---|---|
| 1. Health | 1 | Both artefacts loaded; readiness, not just liveness |
| 2. Happy path | 3 | Correct classification of clearly-worded input |
| 3. Known weakness | 1 | Idiomatic negativity returns `neutral` — documented, not hidden |
| 4. Batch | 2 | Amortised per-item cost; skipped entries counted, never silently dropped |
| 5. Edge cases → 200 | 4 | Emoji, out-of-vocabulary, non-Latin script, single character |
| 6. Malformed → 422 | 7 | Empty, whitespace-only, missing field, wrong name, wrong type, null, empty batch |
| 7. Observability | 1 | `/metrics` from the prediction log |

Verified result: **19 passed, 0 failed.**

## The distinction the collection is built around

| Status | Meaning |
|---|---|
| **422** | The client sent something invalid — *the service is working* |
| **500** | The service broke — *the service is not working* |

Groups 5 and 6 are deliberately adjacent. Group 5 sends input that is *unusual*
(emoji-only, unknown words, Devanagari script) and must return **200** — unusual
is not invalid, and an all-zero feature vector yielding the majority class is an
acceptable answer. Group 6 sends input that is *invalid* and must return **422**,
with the response body naming the failing field and the rule it failed.

No call in the collection may return 500. That is the actual assertion.

## Group 3 is intentional

```
POST /predict/sentiment  {"text":"this film was a complete waste of time"}
→ 200  predicted: neutral  confidence: 0.5185
```

An unambiguously negative input, misclassified. It is in the collection rather
than replaced with an easier example because the diagnosis is informative: the
model over-predicts the majority class (`neutral` at 49.1% against a true rate of
40.5%), and "waste of time" contains no individually negative token for TF-IDF to
weight. See `docs/DEPLOYMENT.md` §5.

## Importing into Postman

If a Postman collection is preferred, each `call` line in the script maps
directly: method, path, and JSON body are all literal. There are no
environment variables beyond `BASE`, and no authentication.
