# Text Classification ML System

**Machine Learning Engineering (PCAM ZC412 / S2-25_PCAMZG412) — Mini-Project I**
BITS Pilani WILP · PGCP in Artificial Intelligence & Machine Learning
Narayan Prasad · 2025paml567 · Group 37 · **Flavor C**

An end-to-end machine learning system that classifies short text by sentiment
and serves it as a monitored REST API — versioned data, tracked experiments, a
packaged service, and drift detection with a documented retraining trigger.

**Start here:** [`SUBMISSION.md`](SUBMISSION.md) maps every deliverable to the
artefacts that satisfy it.

---

## Architecture

```
 ┌─────────────────────────────────────────────────────────────────────┐
 │  RAW DATA                                                      [M2] │
 │  train.csv 27,481 rows · test.csv 4,815 rows                        │
 │  DVC-tracked · Git tag v1.0-raw · pointers in Git, bytes in DVC     │
 └───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  VALIDATION GATE                                               [M2] │
 │  L1 schema → L2 range/domain → L3 statistical → L4 business rule    │
 │  exit(1) on failure — downstream stages never run                   │
 │  BR-01 fabricated column · BR-03 skew vector · BR-04 null text      │
 └───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  INGESTION + CLEANING                                          [M2] │
 │  27,481 → 27,480 rows · every dropped row logged, none silent       │
 │  7 columns excluded: 1 leakage + 6 fabricated metadata              │
 └───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  SHARED FEATURE MODULE            ◄── the load-bearing component    │
 │  split (80/20 stratified, seed 42) THEN fit — never the reverse     │
 │  TF-IDF 1–2 grams · 20,000 features · vocabulary 20,000             │
 │  ► the fitted vectorizer IS the training–serving contract           │
 │  ► exposes transform(), deliberately NO fit()                       │
 └───────────────┬────────────────────────────────┬────────────────────┘
                 │                                │
                 ▼                                ▼
 ┌──────────────────────────────┐   ┌─────────────────────────────────┐
 │  FEATURE STORE (SQLite) [M2] │   │  models/feature_bundle.joblib   │
 │  features_train/features_test│   │  models/classifier.joblib       │
 └──────────────┬───────────────┘   └───────────────┬─────────────────┘
                ▼                                   │
 ┌────────────────────────────────────────────┐     │
 │  TRACKED EXPERIMENTS (MLflow)         [M3] │     │
 │  3 runs · one variable each · hypothesis   │     │
 │  recorded BEFORE execution                 │     │
 │  bootstrap → all runs indistinguishable    │     │
 │  reproducibility audit → delta 0.00e+00    │     │
 │  registry: None → Staging → [gate] → Prod  │     │
 └────────────────────┬───────────────────────┘     │
                      ▼                             ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  SERVING (FastAPI + Pydantic + Docker)                         [M4] │
 │  artefacts loaded ONCE at startup (396 ms) — not per request        │
 │  GET /health · POST /predict/sentiment · POST /predict/batch        │
 │  GET /metrics · GET /docs                                           │
 │  malformed input → 422, never 500 · p50 0.561 ms · 1,445 req/s      │
 └───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  PREDICTION LOG (SQLite)                                       [M5] │
 │  raw text + label + confidence + model version + latency            │
 │  raw text, not vectors — a vector cannot be un-vectorised           │
 └───────────────────────────────┬─────────────────────────────────────┘
                                 ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  DRIFT MONITOR                                                 [M5] │
 │  OOV rate · token chi² · confidence KS + effect floor · length σ    │
 │  ALL COMPUTED WITHOUT LABELS — labels are unavailable in production │
 │  trigger: ≥2 signals AND ≥500 samples → RETRAIN                     │
 │           ≥2 signals AND <500 samples → HOLD                        │
 │  promotion to Production requires named human approval              │
 └─────────────────────────────────────────────────────────────────────┘
```

The whole pipeline is declared in [`dvc.yaml`](dvc.yaml) — seven stages, one
command:

```bash
dvc repro
```

---

## Quick start

```bash
conda env create -f environment.yml
conda activate mle

dvc pull                    # fetch the versioned data
dvc repro                   # validate → ingest → features → train → monitor
pytest tests/ -q            # 57 passed

python -m src.serving.export_model
uvicorn src.serving.app:app --port 8000
```

Then open http://127.0.0.1:8000/docs, or run the full request/response suite:

```bash
bash api_tests/curl_collection.sh        # 19 passed, 0 failed
```

---

## Results

| | |
|---|---|
| **Task** | 3-class sentiment (negative / neutral / positive) |
| **Corpus** | Tweet Sentiment Extraction — 21,984 train / 5,496 held out |
| **Selected model** | TF-IDF + Logistic Regression, `C=1.0` |
| **f1_macro** | 0.6821 |
| **Accuracy** | 0.6807 |
| **Inference p50 / p95** | 0.561 ms / 1.103 ms |
| **Throughput** | ~1,445 req/s per worker |
| **Baseline OOV rate** | 0.0914 — the M5 drift reference |
| **Tests** | 57 passing |

**0.68 is moderate, and that is expected.** Three classes make random guessing
≈0.33. The `neutral` class is intrinsically ambiguous — human annotators
disagree on whether a flat statement is neutral or mildly negative — and tweets
are short, median 64 characters. Published classical results on this dataset
cluster in the 0.68–0.72 range. **The ceiling here is set by the data, not the
algorithm.**

---

## Design decisions

### The artefact is not the model

It is the model **and** the fitted vectorizer, frozen together. A TF-IDF vector
is meaningless without the vocabulary that produced it: token index 4,182 means
one specific word only in the vocabulary that assigned it. Re-fit the vectorizer
and the model's coefficients apply to the wrong features — silently, HTTP 200
throughout.

`FeatureBundle` therefore exposes `transform()` and **no `fit()`**. The serving
path is structurally incapable of re-fitting, and `tests/test_features.py`
asserts that property rather than trusting it.

### A serving constraint decided a modelling choice

The bootstrap found **all three runs statistically indistinguishable** — no
pairwise confidence interval excludes zero. LinearSVC is, on the evidence, as
good a classifier as the baseline.

It was rejected anyway, because it has no `predict_proba` — which breaks the
serving layer's low-confidence flag and blinds the M5 confidence drift signal.

**The best classifier that cannot be monitored is worse than an equivalent
classifier that can be.** Full reasoning in
[`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md).

### The monitor must be able to stay silent

A detector that fires on clean traffic is worse than no detector: it trains the
operator to ignore alerts, so the real alert is ignored too. The control window
— a resample of the baseline pool, undrifted by construction — raises **0 of 4**
signals. That silence is what makes the alerts on the drifted windows mean
something.

### Detection and action are different problems

Window W4 breaches three signals on 120 samples, and the decision is **HOLD**,
not RETRAIN. Retraining on 120 examples would fit the drift's sampling noise
rather than correct for it.

---

## Findings

Three things this project discovered rather than assumed.

**A candidate corpus was rejected on evidence.** The dataset archive contained a
1.6M-row Sentiment140 export truncated at 1,048,572 rows — the Excel worksheet
ceiling of 2²⁰ — with its positive class cut from 800,000 to 248,576. The file
loads without error and **passes schema validation**. Only Level 3 distribution
checking caught it.
→ [`docs/DATA_QUALITY_INCIDENT.md`](docs/DATA_QUALITY_INCIDENT.md) §2

**A fabricated feature column was detected automatically.** `Country` repeats on
a fixed 195-row cycle — verified across 585 consecutive positions — proving it
was appended by row position rather than joined by key. Business rule BR-01
encodes the detection; six columns were excluded as a result.
→ [`docs/DATA_QUALITY_INCIDENT.md`](docs/DATA_QUALITY_INCIDENT.md) §3

**The same statistical error was caught twice, in two modules.** In M3 a 0.005
f1_macro gap proved statistically undetectable; in M5 a confidence shift of
0.009 was significant at p=0.0012 and operationally meaningless. Both stem from
treating a p-value as a decision. The remedies differ — a bootstrap interval in
M3, an effect-size floor in M5 — but it is one recurring failure mode, not two
unrelated bugs.

---

## Repository layout

```
configs/            Task configuration — the pipeline's only variable input
  sentiment.yaml      PRIMARY task
  tickets.yaml        SECONDARY — committed unimplemented, as a falsifiable
                      test of the claim that the pipeline is task-agnostic
data/
  raw/              Immutable source (DVC-tracked, git-ignored)
  processed/        Feature store + prediction log
src/
  ingest/           M2 — loading, cleaning, audit logging
  validation/       M2 — four-level validation framework
  features/         M2 — SHARED module: training AND serving import from here
  training/         M3 — experiments, significance, reproducibility, registry
  serving/          M4 — FastAPI service, model export, benchmark
  monitoring/       M5 — drift signals, simulation
tests/              57 tests, plus the validation negative test
api_tests/          Deliverable 3 — 19 runnable request/response calls
docs/               Incident report, selection record, deployment, monitoring
reports/            Generated JSON evidence — validation, drift, audits
models/             Model + vectorizer bundle (DVC-tracked)
```

---

## Documentation

| Document | Covers |
|---|---|
| [`SUBMISSION.md`](SUBMISSION.md) | All five deliverables mapped to artefacts |
| [`docs/DATA_QUALITY_INCIDENT.md`](docs/DATA_QUALITY_INCIDENT.md) | Rejected corpus, fabricated columns, skew vector |
| [`docs/MODEL_SELECTION.md`](docs/MODEL_SELECTION.md) | Why this run, and why not each other |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | Serving, validation, latency, containerisation |
| [`docs/MONITORING.md`](docs/MONITORING.md) | Drift signals, thresholds, retraining trigger |
| [`docs/screenshots/README.md`](docs/screenshots/README.md) | Evidence index, by rubric line |

---

## Governance record

```
System owner        : Narayan Prasad (2025paml567)
Retraining owner    : same — single-member group
Model in Staging    : sentiment-classifier v1
Source run          : run1_baseline_logreg
Data version        : v1.0-raw
Code version        : recorded as the git_commit tag on every MLflow run
Selection record    : docs/MODEL_SELECTION.md
Promotion gate      : M4 contract tests pass + M5 baseline established
                      + named human approval. Not automatic.
```

### Checklist

```
[ x ] Raw data schema validated before any downstream stage
[ x ] Validation fails loudly (exit 1) rather than warning and continuing
[ x ] Validation gate proven to fail, via injected-defect negative test
[ x ] Data quality findings documented with evidence, not merely fixed
[ x ] Fabricated and leaking columns excluded with recorded justification
[ x ] Environment pinned via conda environment.yml
[ x ] Dataset versioned and tagged with DVC (v1.0-raw)
[ x ] Feature logic centralised — not duplicated in training and serving
[ x ] All experiments logged in MLflow with full parameters
[ x ] Winning run reproducible from logged configuration (delta 0.00e+00)
[ x ] Model selection justified, including why each alternative was rejected
[ x ] Inference API has Pydantic input validation
[ x ] Malformed input returns 422 with the failing field named
[ x ] Model artefact frozen and immutable after selection
[ x ] All predictions logged with inputs, outputs, model version, latency
[ x ] Drift detection produces actionable output
[ x ] Drift detector proven NOT to fire on undrifted traffic
[ x ] Retraining trigger designed with justified thresholds
[ x ] Docker containerisation (built in CI)
[ x ] Unit tests on the feature pipeline (17)
[ x ] API contract tests (34)
[ x ] Automated pipeline — GitHub Actions on every push
[   ] Label feedback loop for retraining data
[   ] Thresholds tuned against observed production false-positive rates
[   ] Authentication on the inference endpoint
```

The last three are open deliberately, and are recorded as limitations rather
than left implied.

---

## Residual risks

- **The drift is synthetic.** Real drift needs production traffic collected over
  months. Constructed drift is cleaner and more abrupt than the real thing, so
  these windows test the detector's wiring — they are not a claim about
  real-world sensitivity.
- **The model over-predicts the majority class.** `neutral` is predicted 49.1%
  of the time against a true rate of 40.5%. Diagnosed in
  [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) §5; `class_weight='balanced'` is
  the queued experiment.
- **Label accuracy is unverifiable.** No automated check exists for label
  correctness. A sampled audit was out of scope.
- **SQLite is not production-grade** under concurrent writers, for either the
  feature store or the prediction log. Chosen for scope; the limitation is
  architectural, not accidental.
- **No authentication.** Production would require an API key or OIDC at the
  gateway.

---

## References

- **T1** Crowe, R. et al. *Machine Learning Production Systems.* O'Reilly, 2024.
- **T2** Burkov, A. *Machine Learning Engineering.* 2020.
- **R1** McMahon, A.P. *Machine Learning Engineering with Python*, 2nd ed. Packt, 2023.
- Sculley, D. et al. *Hidden Technical Debt in Machine Learning Systems.* NeurIPS 2015.
- Dataset: Tweet Sentiment Extraction (Kaggle). Sentiment140 (Go, Bhayani & Huang, 2009) — evaluated and rejected; see the incident report.
