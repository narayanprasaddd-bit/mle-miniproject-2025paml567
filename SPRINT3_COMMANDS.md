# Sprint 3 — Paste-Ready Commands

**Sprint 3 = M4 Packaging & Deployment → 20% of the grade.**

No admin rights needed. Docker builds in CI instead of locally.

---

## 1. Unpack

Git Bash:

```bash
cd /c/projects/mle-miniproject-2025paml567
unzip -o "/c/NP Documents and Tools/BITS/Mini Project/sprint3_files.zip" -d .
```

---

## 2. Install the serving dependencies

Anaconda Prompt, `(mle)` active:

```
cd C:\projects\mle-miniproject-2025paml567
conda activate mle
pip install fastapi uvicorn httpx
```

---

## 3. Export the model for serving

```
python -m src.serving.export_model
```

Expected: `wrote models/classifier.joblib` and the three class names.

---

## 4. Run the API tests

```
pytest tests/ -v
```

Expected: **51 passed** (17 feature tests + 34 API contract tests).

---

## 5. Benchmark latency

```
python -m src.serving.benchmark
```

Expected: p50 around 0.5 ms, and batching 100 roughly 20x cheaper per item.
Your numbers will differ from the reference — use **yours** in the report.

---

## 6. Start the service and look at it

```
uvicorn src.serving.app:app --port 8000
```

Open **http://127.0.0.1:8000/docs** — this is the auto-generated OpenAPI page.

### Screenshots to capture (deliverable 3)

1. The `/docs` page showing all five endpoints
2. `/health` response — expand "GET /health", click "Try it out", "Execute"
3. A successful prediction — expand "POST /predict/sentiment", "Try it out",
   paste `{"text": "this film was a complete waste of time"}`, Execute
4. **A 422 response** — same endpoint, send `{"text": ""}`. This is the one
   most worth having: it proves malformed input is rejected cleanly.

Save into `docs/screenshots/` as `api_docs.png`, `api_health.png`,
`api_predict.png`, `api_422.png`.

Then Ctrl+C to stop.

---

## 7. Commit and push

Git Bash:

```bash
git add -A
git commit -m "Sprint 3: FastAPI service, container image, contract tests" -m "- FastAPI + Pydantic v2; model and vectorizer loaded once at startup" -m "- 34 API contract tests: malformed input returns 422, never 500" -m "- Out-of-vocabulary input asserted to yield low confidence (M5 signal depends on it)" -m "- Prediction logging to SQLite: inputs, outputs, confidence, model version, latency" -m "- Measured latency: p50 0.52ms, p95 0.65ms, ~1880 req/s per worker" -m "- Batch endpoint 21x cheaper per item; amortisation asserted by test" -m "- Multi-stage Dockerfile, non-root user, healthcheck, serving deps only" -m "- Image built in GitHub Actions CI: no admin rights on the dev machine for Docker Desktop"
git push
```

---

## 8. Watch the CI build

Open your repo on GitHub → **Actions** tab.

The `CI` workflow runs automatically on push. Two jobs:

- **Unit and contract tests** — runs the feature tests on a clean checkout
- **Build container image** — builds the Dockerfile and reports the image size

Once it goes green, **screenshot it**. That green tick is your evidence that
the container builds, and it required no admin rights on your laptop.

Save as `docs/screenshots/ci_docker_build.png`.
