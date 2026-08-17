# Submission Checklist

**Course:** Machine Learning Engineering (PCAM ZC412 / S2-25_PCAMZG412)
**Assessment:** Mini-Project-I — EC-1, 40% of course grade
**Student:** Narayan Prasad · 2025paml567 · Group 37
**Flavor:** C — Support Ticket / Review Sentiment Classifier
**Due:** 24 August 2026, 4:06 PM

This document maps each of the brief's five deliverables to the specific
artefacts that satisfy it, so nothing has to be hunted for.

---

## Deliverable 1 — Versioned dataset and pipeline code

> *"GitHub/GitLab repository link, with commit history reflecting weekly progress."*

**Repository:** `https://github.com/narayanprasaddd-bit/mle-miniproject-2025paml567`

| Evidence | Location |
|---|---|
| Dataset versioned with DVC | `data/raw/train.csv.dvc`, `data/raw/test.csv.dvc` |
| Dataset version tag | Git tag `v1.0-raw` |
| Declarative pipeline | `dvc.yaml` — 7 stages, plus `dvc.lock` |
| Reproducible in one command | `dvc repro` |
| Commit history | 5 sprint-labelled commits |

**On the commit history.** Commits are labelled `Sprint 0` … `Sprint 4`, each a
coherent unit of work with a body listing the artefacts it introduced. The
repository was **not** a single upload: `bc6f40d` → `cfb4809` → `f63b479` → …
each builds on the last, and the MLflow runs carry the git commit hash of the
code that produced them.

**On what is and is not in Git.** The raw CSVs are DVC-tracked and deliberately
absent from Git; the `.dvc` pointer files carry their hashes and *are* committed.
That separation is the graded distinction, and it is verifiable:

```bash
git ls-files data/raw/          # only .dvc pointers and .gitkeep
cat data/raw/train.csv.dvc      # a hash and a size, not CSV content
```

---

## Deliverable 2 — Experiment tracking logs and model comparison report

> *"Screenshots or exported logs, e.g., MLflow."*

Provided in **both** forms — machine-readable JSON as the primary record, plus
screenshots for the human-readable view.

| Evidence | Location |
|---|---|
| **Model comparison report** | `docs/MODEL_SELECTION.md` |
| Exported run comparison | `reports/model_comparison.json` |
| Bootstrap significance test | `reports/significance_test.json` |
| Reproducibility audit | `reports/reproducibility_audit.json` |
| Per-run classification reports | `reports/classification_report_run*.txt` |
| MLflow UI screenshots | `docs/screenshots/mlflow_*.png` |

**Three tracked runs**, one variable changed per run, each with its hypothesis
recorded *before* execution:

| Run | Varied | f1_macro | Accuracy |
|---|---|---|---|
| **run1_baseline_logreg** ← selected | — (baseline) | **0.6821** | 0.6807 |
| run2_linearsvc | algorithm | 0.6739 | 0.6705 |
| run3_logreg_C0.5 | regularisation | 0.6773 | 0.6776 |

**Reproducibility demonstrated, not asserted.** The selected run was rebuilt from
its logged MLflow parameters alone — no access to the training script's in-memory
objects — retrained, and re-scored. Metrics matched to `0.00e+00`.

**Selection was not by score.** A 1,000-sample paired bootstrap found that
**none** of the three runs is statistically distinguishable from the others —
every pairwise 95% confidence interval spans zero. The ordering is stable but the
gaps are not real.

That result rules out an entire line of reasoning, and forces the decision onto
other grounds:

- **run3** was rejected on **simplicity** — `C=1.0` is the scikit-learn default
  and a non-default value must earn its place, which a statistically invisible
  difference does not.
- **run2 (LinearSVC)** is on the evidence *as good a classifier as the baseline*,
  and was rejected anyway because it has no `predict_proba` — which breaks the
  serving layer's low-confidence flag and blinds the M5 confidence drift signal.

**A modelling choice was therefore decided by a serving constraint.** That
ordering is the point: the best classifier that cannot be monitored is worse than
an equivalent classifier that can be. Full reasoning in
`docs/MODEL_SELECTION.md` §3–§5.

---

## Deliverable 3 — Deployed model with a working API endpoint

> *"Including sample request/response test calls (e.g., Postman collection or curl commands)."*

| Evidence | Location |
|---|---|
| **Runnable curl collection — 19 calls** | `api_tests/curl_collection.sh` |
| Service implementation | `src/serving/app.py` |
| Container image definition | `Dockerfile`, `.dockerignore` |
| CI build of the image | `.github/workflows/ci.yml` |
| Deployment record | `docs/DEPLOYMENT.md` |
| Contract tests (34) | `tests/test_api.py` |
| Latency measurements | `reports/latency_benchmark.json` |
| API screenshots | `docs/screenshots/0*_api_*.png` |

**Run the collection:**

```bash
uvicorn src.serving.app:app --port 8000        # terminal 1
bash api_tests/curl_collection.sh              # terminal 2
```

Verified result: **19 passed, 0 failed.** The collection is ordered as a
narrative — health, correct predictions, a documented weakness, batching, edge
cases that must succeed, malformed input that must be rejected, then
observability.

**Endpoints:**

```
GET  /health              liveness + readiness, reports BOTH artefacts
POST /predict/sentiment   label + per-class probabilities + confidence
POST /predict/batch       up to 100 texts, amortised
GET  /metrics             volume, latency, low-confidence rate
GET  /docs                OpenAPI schema
```

**Measured performance:** p50 0.561 ms · p95 1.103 ms · ~1,445 req/s per worker ·
batching 100 is 30× cheaper per item.

**On containerisation.** The development machine is a managed corporate laptop
without administrator rights, so Docker Desktop — which requires the WSL2 backend
and an elevated installer — could not be installed. The image is built on
GitHub's Linux runners instead, on every push. Rationale in
`docs/DEPLOYMENT.md` §8.

---

## Deliverable 4 — Monitoring log, drift simulation, retraining trigger

| Evidence | Location |
|---|---|
| **Monitoring and retraining design** | `docs/MONITORING.md` |
| Drift report — 5 windows | `reports/drift_report.json` |
| Drift detector | `src/monitoring/drift.py` |
| Drift simulation | `src/monitoring/simulate.py` |
| Prediction log (live traffic) | `data/processed/predictions.db` |
| Drift tests (6) | `tests/test_drift.py` |
| Screenshots | `docs/screenshots/drift_*.png` |

**Four label-free signals.** In production the true labels are unavailable at
inference time — which is the whole reason drift detection exists. Every signal
is therefore computed without ground truth:

| Signal | Test | Threshold |
|---|---|---|
| OOV rate | Proportion | > 0.15 |
| Token frequency | Chi-squared | p < 0.01 |
| Confidence | KS **+ effect floor** | p < 0.01 and abs(delta) >= 0.02 |
| Document length | Sigma shift | > 1σ investigate, > 2σ halt |

**Five windows, including a control:**

| Window | Breached | Decision |
|---|---|---|
| **W0 control (no drift)** | **0 of 4** | NO_ACTION |
| W1 vocabulary drift | 3 of 4 | RETRAIN |
| W2 topic drift | 3 of 4 | RETRAIN |
| W3 format drift | 2 of 4 | RETRAIN |
| W4 drift, 120 samples | 3 of 4 | **HOLD** |

**W0 is the primary result.** A monitor that fires on clean traffic is worse than
no monitor — it trains the operator to ignore alerts, so the real alert is
ignored too. W0 is a resample of the baseline pool, undrifted by construction,
and raises nothing.

**Retraining trigger:** ≥2 signals breached **and** ≥500 new samples. One signal
is noise; two is corroboration. W4 demonstrates the HOLD path — the drift is real
but retraining on 120 examples would fit the drift's noise rather than correct
for it. Promotion to Production requires named human approval and is not
automatic.

---

## Deliverable 5 — README, architecture diagram, demo

| Evidence | Location |
|---|---|
| README with architecture diagram | `README.md` |
| Governance record | `README.md` |
| Screenshot index | `docs/screenshots/README.md` |
| Data quality incident report | `docs/DATA_QUALITY_INCIDENT.md` |
| Recorded demo (5–7 min) | *pending — script in `docs/DEMO_SCRIPT.md`* |

---

## Test suite

```bash
pytest tests/ -q        # 57 passed
```

| Suite | Count | Covers |
|---|---|---|
| `tests/test_features.py` | 17 | Shared feature module, training–serving skew |
| `tests/test_api.py` | 34 | API contract, malformed input, edge cases |
| `tests/test_drift.py` | 6 | Drift detection, false-positive control |
| `tests/inject_defects.py` | — | Negative test for the validation gate |

---

## Rubric coverage

| Criterion | Weight | Primary evidence |
|---|---|---|
| Data Engineering & Versioning (M2) | 20% | `src/validation/`, `src/ingest/`, `src/features/`, `dvc.yaml`, `docs/DATA_QUALITY_INCIDENT.md` |
| Experimentation & Reproducibility (M3) | 20% | `src/training/`, `docs/MODEL_SELECTION.md`, `reports/reproducibility_audit.json` |
| Model Packaging & Deployment (M4) | 20% | `src/serving/`, `Dockerfile`, `api_tests/`, `docs/DEPLOYMENT.md` |
| Monitoring, Drift & Retraining (M5) | 20% | `src/monitoring/`, `docs/MONITORING.md`, `reports/drift_report.json` |
| Documentation & Presentation | 20% | `README.md`, `docs/`, this file, the demo recording |

---

## Three things worth the grader's attention

**1. A candidate corpus was rejected on evidence.** The dataset archive contained
a 1.6M-row Sentiment140 export truncated at 1,048,572 rows — the Excel worksheet
ceiling of 2²⁰ — reducing its positive class from 800,000 to 248,576. The file
loads without error and passes schema validation; only Level 3 distribution
checking caught it. `docs/DATA_QUALITY_INCIDENT.md` §2.

**2. A fabricated feature column was detected automatically.** The `Country`
column repeats on a fixed 195-row cycle — verified across 585 consecutive
positions — proving it was appended by row position rather than joined by key.
Business rule BR-01 in `src/validation/schema.py` encodes the detection; six
columns were excluded as a result.

**3. The same statistical error was caught twice, in two modules.** In M3 a 0.002
f1_macro difference between runs was statistically detectable and operationally
irrelevant; in M5 a confidence shift of 0.009 was significant at p=0.0012 and
equally meaningless. Both stem from treating a p-value as a decision. The
remedies differ — a bootstrap confidence interval in M3, an effect-size floor in
M5 — but recognising it as one recurring failure mode rather than two unrelated
bugs is the point.
