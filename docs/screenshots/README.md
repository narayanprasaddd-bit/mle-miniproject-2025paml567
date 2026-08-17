# Evidence Screenshots

Index of visual evidence, mapped to the rubric line each item supports.

The brief's Deliverables section asks for experiment tracking logs
("screenshots or exported logs, e.g. MLflow") and a monitoring/drift report.
Every screenshot below is paired with a machine-readable JSON artefact in
`reports/`, so the evidence exists in both forms — the JSONs are the primary
record and are version-controlled; the screenshots are the human-readable view.

---

## M2 — Data Engineering & Versioning (20%)

| File | Shows | Paired artefact |
|---|---|---|
| `validation_pass.png` | Clean data: L1/L2 PASS, L3 PASS, three L4 findings, exit 0 | `reports/validation_report.json` |
| `validation_fail.png` | Injected defects: **five blocking failures, pipeline halted**, exit 1 | `reports/validation_report.json` |
| `dvc_repro.png` | `dvc repro` running validate → ingest → build_features in dependency order | `dvc.lock` |
| `dvc_dag.png` | The pipeline dependency graph | `dvc.yaml` |

**These two must be read as a pair.** A validation suite that has only ever run
against clean data demonstrates that nothing was detected, not that detection
works. `validation_fail.png` is the negative test: three defects were
deliberately injected (an invalid category, an invalid enum value, a duplicated
key) and the gate caught all three across two levels, then exited 1.

**The FAIL screenshot is the more valuable of the two.** M2 §2.5 frames it
directly: *"a validation failure is a success, not a system error."* The exit
code 1 is the pipeline working correctly — refusing to let corrupt data reach
feature engineering.

**What to look for in `validation_pass.png`** — the three Level 4 business-rule
findings, which are the project's own discoveries rather than generic checks:

- **BR-01** — the `Country` column repeats on a fixed 195-row cycle across 585
  compared positions, proving it was appended by row position rather than joined
  by key. Six columns excluded as a result.
- **BR-03** — 39.9% of `text` values carry leading whitespace, a
  training–serving skew vector.
- **BR-04** — one null `text`, dropped and logged rather than silently filtered.

Full evidence chain in `docs/DATA_QUALITY_INCIDENT.md`, which also records the
candidate corpus that was **rejected**: a 1.6M-row Sentiment140 export found to
be truncated at 1,048,572 rows — the Excel worksheet ceiling of 2²⁰ — with its
positive class reduced from 800,000 to 248,576. That defect is invisible to
schema validation and was caught only by Level 3 distribution checking.

**Reproduce both:**

```bash
# PASS path
python -m src.validation.validate --config configs/sentiment.yaml

# FAIL path
python -m tests.inject_defects
python -m src.validation.validate --config configs/sentiment.yaml \
    --input data/raw/_corrupt_test.csv
```

---

## M3 — Experimentation & Reproducibility (20%)

| File | Shows | Paired artefact |
|---|---|---|
| `mlflow_runs_table.png` | Three tracked runs with `accuracy` and `f1_macro` side by side | `reports/model_comparison.json` |
| `mlflow_run_detail.png` | run1 parameters (16), metrics (5), and reproducibility tags: `git_commit`, `data_version_tag`, `hypothesis` | — |
| `mlflow_model_registry.png` | `sentiment-classifier` v1 with the `@staging` alias | — |

**What to look for in `mlflow_run_detail.png`:** the `hypothesis` tag. It records
what the run was expected to show *before* it ran, which is the M3 §3.3
discipline — a prediction stated in advance, not a result rationalised
afterwards.

---

## M4 — Packaging & Deployment (20%)

| File | Shows | Paired artefact |
|---|---|---|
| `01_api_docs.png` | OpenAPI page, four endpoints, generated request/response schemas | — |
| `02_api_health.png` | 200 — `model_loaded` **and** `vectorizer_loaded` both true, 20,000 features, three classes | — |
| `03_api_predict_request.png` | Request body submitted to `/predict/sentiment` | — |
| `04_api_predict_response.png` | 200 — label, per-class probabilities, confidence, latency, model version | `reports/latency_benchmark.json` |
| `05_api_422_request.png` | Empty string submitted as input | — |
| `06_api_422_response.png` | **422 `string_too_short`** — rejected before reaching model code | — |
| `ci_docker_build.png` | GitHub Actions run building the container image | — |

**The 422 pair is the direct evidence** for the brief's M4 task, "handle
malformed/edge-case inputs." Note that the response body names the failing field
and the rule that failed rather than returning an opaque error.

**`02_api_health.png` reports both artefacts separately** because a service with
a model but no vectorizer would start cleanly and then produce silent nonsense.
See `docs/DEPLOYMENT.md` §2.

**`04_api_predict_response.png` contains a misclassification** — an
unambiguously negative input returned `neutral` at 0.5185 confidence. This is
deliberately retained rather than re-shot with an easier example. The diagnosis
is in `docs/DEPLOYMENT.md` §5: the model over-predicts the majority class
(neutral predicted 49.1% against a true rate of 40.5%).

**`ci_docker_build.png` substitutes for a local `docker build`.** The
development machine is a managed corporate laptop without administrator rights,
so Docker Desktop — which needs the WSL2 backend and an elevated installer —
could not be installed. The image is built on GitHub's Linux runners instead.
Rationale in `docs/DEPLOYMENT.md` §8.

---

## M5 — Monitoring, Drift & Retraining (20%)

| File | Shows | Paired artefact |
|---|---|---|
| `drift_detection_1.png` | **W0 control: 0 of 4 breached, NO_ACTION** · W1 vocabulary drift: 3 of 4, RETRAIN | `reports/drift_report.json` |
| `drift_detection_2.png` | W2 topic drift: 3 of 4, RETRAIN · W3 format drift | `reports/drift_report.json` |
| `drift_detection_3.png` | W3 format drift detail · **W4: HOLD** on insufficient samples | `reports/drift_report.json` |
| `drift_tests.png` | Six drift tests passing | — |

**`drift_detection_1.png` is the most important screenshot in the project.**
W0 is a resample of the baseline pool — undrifted by construction — and fires
zero signals. A monitor that fires on clean traffic is worse than no monitor: it
trains the operator to ignore alerts, so the real alert is ignored too.
Demonstrating silence is the primary result; the alerts on W1–W4 only mean
something because of it.

**In `drift_detection_2.png`, note W3's two `[ ok ]` lines:**

- `oov_rate 0.0889` — vocabulary unchanged, because format drift concatenates
  *existing* texts. The signals discriminate between *kinds* of drift, not
  merely detect that something changed.
- `confidence_ks 0.001201` — flagged "statistically significant but below the
  0.02 effect floor (delta -0.009) — not actioned." A KS test on large samples
  detects shifts far too small to act on. See `docs/MONITORING.md` §5.

**In `drift_detection_3.png`, W4 shows HOLD** — three signals breached on 120
samples. The drift is real but there is not enough data to retrain on.
Detecting drift and being able to act on it are different problems.

**`drift_tests.png` is worth reading as a summary of the design.** The six test
names state what the monitor does and what it deliberately does not do:

```
test_control_window_raises_no_alert
test_vocabulary_drift_is_detected
test_topic_drift_is_detected
test_format_drift_fires_length_but_not_oov
test_insufficient_samples_holds_rather_than_retrains
test_confidence_signal_respects_the_effect_floor
```

---

## Test suite

| File | Shows |
|---|---|
| `test_suite_all.png` | **57 passed** — 17 feature + 34 API + 6 drift |

---

## Sprint-to-rubric mapping

The sections above are organised by **rubric line**, since that is what the
marker assesses against. For cross-reference with the commit history:

| Commit | Sprint | Rubric line |
|---|---|---|
| `bc6f40d` | Sprint 0 | M2 — scaffold, validation framework |
| `cfb4809` | Sprint 1 | M2 — DVC, ingestion, shared feature module |
| `f63b479` | Sprint 2 | M3 — MLflow, significance testing, registry |
| — | Sprint 3 | M4 — FastAPI, Docker, contract tests |
| — | Sprint 4 | M5 — drift signals, simulation, retraining trigger |

---

## Naming convention

- M4 API screenshots are numbered `01`–`06` because they form an ordered
  walkthrough of one session.
- All other files are named by content.
- Sequential suffixes (`_1`, `_2`, `_3`) indicate a single console output split
  across captures because it exceeded one screen.
