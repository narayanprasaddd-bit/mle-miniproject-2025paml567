# Deployment Record

**Module reference:** M4 — Model Packaging, Deployment & Serving
**Service:** Sentiment Classification Service v1.0.0
**Model version:** `sentiment-v1` (MLflow run `run1_baseline_logreg`, id `ed72d76c`)

---

## 1. What is deployed

```
  POST /predict/sentiment    single text  -> label + probabilities + confidence
  POST /predict/batch        up to 100 texts, amortised
  GET  /health               liveness + readiness, reports BOTH artefacts
  GET  /metrics              volume, latency, low-confidence rate
  GET  /docs                 OpenAPI schema (auto-generated)
```

FastAPI + Pydantic v2, served by Uvicorn, packaged as a multi-stage Docker image.

---

## 2. The artefact is a pair, not a file

The deployed artefact is **two** files that must never be separated:

| File | Role |
|---|---|
| `models/classifier.joblib` | The trained logistic regression |
| `models/feature_bundle.joblib` | The fitted TF-IDF vectorizer — vocabulary and IDF weights |

A TF-IDF vector is meaningless without the vocabulary that produced it. Token
index 4,182 means one specific word only in the vocabulary that assigned it.
Deploy a mismatched pair and the model's coefficients are applied to the wrong
features — with **no error raised**. HTTP 200 throughout, every prediction
subtly wrong.

Two structural defences, not just documentation:

1. `FeatureBundle` exposes `transform()` and **no `fit()`**. The serving path
   is structurally incapable of re-fitting the vectorizer.
2. `/health` reports `model_loaded` and `vectorizer_loaded` separately, so a
   half-loaded service is visible rather than silently wrong.

---

## 3. Why the model is exported rather than pulled from MLflow at runtime

The container does not talk to the MLflow tracking server. If the server is
down, the network is partitioned, or the run is later archived, inference must
keep working.

So the selected model is exported once, at build time
(`python -m src.serving.export_model`), into an immutable artefact the container
carries with it. After selection the model is **frozen** — a new model means a
new artefact and a new deployment, never an in-place mutation of a running
service.

---

## 4. Input validation — 422, never 500

The governing distinction:

| Status | Meaning |
|---|---|
| **422** | The client sent something invalid — *the service is working* |
| **500** | The service broke — *the service is not working* |

Conflating them makes production debugging materially harder: a 500 sends an
engineer looking for a bug that does not exist.

Pydantic validates the request body **before** the handler runs, so malformed
input never reaches model code.

### Rejected with 422

| Input | Why |
|---|---|
| Missing `text` field | Contract violation |
| `text` as int, list, or null | Type violation |
| `""` | `min_length=1` |
| `"   "` | Custom validator — whitespace tokenises to nothing, and a confident label on empty input is worse than a refusal |
| >4,000 characters | `max_length` bound; prevents unbounded work per request |
| Malformed JSON body | Parse failure |
| Batch of >100 or 0 texts | Bounds worst-case request latency |

### Accepted with 200 — unusual is not invalid

Single characters, punctuation-only, emoji-only, entirely out-of-vocabulary
text, accented and non-Latin scripts, and text with surrounding whitespace all
return a valid prediction. An all-zero feature vector yielding the majority
class is an acceptable answer; a stack trace is not.

**34 contract tests** in `tests/test_api.py` assert all of the above. Full suite: **51 passed** (17 feature + 34 API).

One test is worth naming: out-of-vocabulary input must come back with
**confidence < 0.75**. If the model were confident on meaningless input, a
rising OOV rate would not surface as falling confidence, and the M5 drift
signal would be blind.

---

## 5. An observed misclassification, and why it is expected

During manual verification the service was sent an unambiguously negative
phrase and returned the wrong class:

```
input:  "this film was a complete waste of time"
output: predicted = neutral   confidence = 0.5185
        probabilities: negative 0.3054 | neutral 0.5185 | positive 0.1761
```

This is recorded rather than hidden, because the diagnosis is informative.

**It is not a serving defect.** The same input produces the same output when the
model is called directly, so the API layer is faithfully reporting what the
model believes. The failure is in the model, not the deployment.

**The model over-predicts `neutral`.** Comparing the true and predicted label
distributions on the held-out split:

| Class | True share | Predicted share |
|---|---|---|
| neutral | 40.45% | **49.14%** |
| positive | 31.24% | 27.29% |
| negative | 28.31% | 23.56% |

`neutral` is the largest class in training, so it is the cheapest guess whenever
evidence is weak — and a short phrase gives little evidence. The confidence of
0.5185 reflects that: the model is barely committing, and `negative` is the
runner-up at 0.3054.

**The pattern is consistent.** Strongly-worded input classifies correctly and
confidently:

| Input | Predicted | Confidence |
|---|---|---|
| "terrible awful worst movie ever" | negative | 0.850 |
| "i loved it absolutely brilliant" | positive | 0.900 |
| "the film was ok i guess" | neutral | 0.732 |
| "waste of time" | neutral | 0.553 |

The model handles explicit sentiment vocabulary well. It fails on *idiomatic*
negativity — "waste of time" carries no individually negative token, and TF-IDF
sees only word counts. The bigram "waste of" would help, but it must clear
`min_df=2` and survive the 20,000-feature cap to be in the vocabulary at all.

**Three implications recorded for the record:**

1. **The low-confidence flag is doing its job.** At 0.5185 this prediction sits
   just above the 0.50 threshold, so it was not flagged — which argues the
   threshold is set slightly too low. Raising it to 0.55 would route this case
   for human review. That is a tuning decision requiring a labelled sample of
   production traffic, so it is deferred rather than guessed.
2. **`class_weight='balanced'`** is the obvious next experiment. It would
   penalise the majority-class default and is a one-parameter change.
3. **This is what an f1_macro of 0.68 looks like in practice.** Aggregate
   metrics are abstract; a single wrong answer on an obvious input is concrete.
   Both are the same fact.

---

## 6. Measured latency and throughput

Measured, not asserted — `python -m src.serving.benchmark`, 500 trials on real
held-out text.

| Metric | Value |
|---|---|
| Artefact load (once, at startup) | 396 ms |
| p50 single-request inference | **0.561 ms** |
| p95 | 1.103 ms |
| p99 | 1.521 ms |
| max | 1.649 ms |
| Throughput | ~1,445 req/s per worker |

### Batching

| Batch size | Total | Per item |
|---|---|---|
| 1 | 0.81 ms | 0.810 ms |
| 10 | 1.07 ms | 0.107 ms |
| 50 | 1.61 ms | 0.032 ms |
| 100 | 2.67 ms | **0.027 ms** |

Batching 100 is **30× cheaper per item**. Sparse matrix construction and the
matrix multiply both amortise across the batch.

Note that absolute per-item cost converges (0.027 ms at size 100) while the
*speedup ratio* depends on the single-call baseline. The fixed per-call
overhead — Python function dispatch, sparse matrix allocation — is what
batching eliminates, and it is roughly constant regardless of batch size.

**Percentiles, not the mean.** A mean latency hides the tail, and the tail is
what users experience as "the service is slow".

**The 396 ms artefact load is the load-bearing number.** It is **706×** the p50
inference time. Loading per request would make artefact loading 99.86% of the
response, and cap throughput at roughly 2.5 requests per second instead of
1,445 — a 570-fold reduction. This is why loading happens once, in the FastAPI
lifespan handler, and not inside the request handler.

---

## 7. Container design

Multi-stage build, `python:3.11-slim` base.

| Decision | Reason |
|---|---|
| Multi-stage | Build toolchain (gcc, headers) never reaches the runtime image — smaller, and no compiler CVEs to patch |
| Non-root `appuser` | A process that does not need root should not have it. If compromised, the attacker inherits an unprivileged account |
| Serving deps only | No `mlflow`, `dvc`, or `pytest` in the runtime image — development tooling is attack surface |
| `HEALTHCHECK` | Lets an orchestrator decide readiness and restart policy |
| **One worker** | The model loads into each worker's memory, so N workers means N copies of the artefacts. Scale by running more **containers** — memory stays predictable and the orchestrator schedules |
| Layer ordering | Least- to most-frequently-changed, so an unrelated edit does not invalidate the cache |

---

## 8. How the image is built — and an honest constraint

**Constraint.** The development machine is a managed corporate laptop without
administrator rights. Docker Desktop on Windows requires the WSL2 backend and
an elevated installer, so it could not be installed locally.

**Resolution.** The image is built in CI on GitHub's Linux runners
(`.github/workflows/ci.yml`), triggered on every push to `main`.

This is arguably **stronger** evidence than a local build:

- the build log is public and reproducible by anyone;
- it proves the Dockerfile works on a **clean** machine, not one carrying the
  author's incidental local state;
- it runs automatically on every commit, so the Dockerfile cannot silently rot.

It also closes the first of the three items the course tutorial deferred — an
automated pipeline rather than a manual one.

**Note on artefacts in CI.** The model files are DVC-tracked and therefore not
in the Git checkout. The CI job substitutes placeholders so that every
Dockerfile instruction is exercised and verified. A deployable image is built
from a checkout with `dvc pull` run first; the workflow's health-check step
activates automatically when real artefacts are present.

---

## 9. Reproducing locally

```bash
dvc repro                                   # data -> features
python -m src.serving.export_model          # freeze the selected model
uvicorn src.serving.app:app --port 8000     # serve
```

Then open http://127.0.0.1:8000/docs, or:

```bash
curl -X POST http://127.0.0.1:8000/predict/sentiment \
     -H 'Content-Type: application/json' \
     -d '{"text":"this film was a complete waste of time"}'
```

With Docker available:

```bash
docker build -t sentiment-classifier:1.0 .
docker run -p 8000:8000 sentiment-classifier:1.0
```

---

## 10. Residual risks

- **SQLite prediction log is not production-grade** under concurrent writers.
  Chosen deliberately for scope; a real deployment would write to Kafka or a
  managed database. The limitation is architectural, not accidental.
- **No authentication.** The service is unauthenticated by design for this
  exercise. Production would require an API key or OIDC at the gateway.
- **Single worker, single container.** No horizontal scaling or load balancer
  is configured. The design supports it — the service is stateless apart from
  the prediction log — but it is not demonstrated.
- **The model over-predicts the majority class.** See §5. Documented, diagnosed,
  and not yet fixed: `class_weight='balanced'` is the queued experiment.
- **Log writes fail silently.** A failed log write does not fail the request,
  because a monitoring gap is preferable to an outage. The trade-off is that a
  persistent write failure would degrade monitoring quietly.
