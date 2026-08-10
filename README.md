# Text Classification ML System — Mini-Project I

**Course:** Machine Learning Engineering (PCAM ZC412 / S2-25_PCAMZG412)
**Programme:** BITS Pilani WILP — PGCP in Artificial Intelligence & Machine Learning
**Student:** Narayan Prasad · ID 2025paml567 · Group 37
**Flavor:** C — Support Ticket / Review Sentiment Classifier
**Submission:** Mini-Project-I · due 24 August 2026

---

## System Overview

```
Purpose      : Classify short social/support text by sentiment, served as a
               production-style REST API with drift monitoring and a
               documented retraining trigger.
Task         : 3-class classification (negative / neutral / positive)
Corpus       : Tweet Sentiment Extraction — 27,481 train / 4,815 held-out
Prediction   : Class label + per-class probabilities + confidence flag
Consumers    : (simulated) support triage queue, review moderation dashboard
```

**Architectural claim of this repository.** The pipeline is **task-agnostic**.
Every stage reads its behaviour from a config file, so adding a second
classification task requires a new config — not new pipeline code. A second
config (`configs/tickets.yaml`, support-ticket urgency) is committed as the
falsifiable test of that claim.

---

## Repository Layout

```
configs/                  Task configuration — the pipeline's only variable input
  sentiment.yaml            PRIMARY task
  tickets.yaml              SECONDARY task (stretch goal; proves generality)
data/
  raw/                    Immutable source data (DVC-tracked, git-ignored)
  processed/              Engineered features (DVC-tracked, git-ignored)
src/
  ingest/                 M2 — source loading, cleaning, audit logging
  validation/             M2 — four-level validation framework
    schema.py               Pandera contracts + business rules BR-01..BR-04
    validate.py             Runnable gate; exits 1 on blocking failure
  features/               M2 — SHARED feature module (training + serving)
  training/               M3 — MLflow-tracked experiments
  serving/                M4 — FastAPI inference service
  monitoring/             M5 — prediction logging, drift signals
tests/
  inject_defects.py       Negative test: proves the validation gate fails
docs/
  DATA_QUALITY_INCIDENT.md  Findings from ingestion-boundary validation
reports/                  Generated validation / drift / comparison reports
models/                   Serialised model + vectorizer bundle (DVC-tracked)
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate mle
```

Place the source data in `data/raw/`:

```
data/raw/train.csv        Tweet Sentiment Extraction — training split
data/raw/test.csv         Tweet Sentiment Extraction — held-out split
```

Validate before anything else runs:

```bash
python -m src.validation.validate --config configs/sentiment.yaml
```

---

## Pipeline Components

| Component | Module | Entry point | Status |
|---|---|---|---|
| Data validation (L1–L4) | M2 | `src/validation/validate.py` | ✅ Complete |
| Data quality incident report | M2 | `docs/DATA_QUALITY_INCIDENT.md` | ✅ Complete |
| Validation negative test | M2 | `tests/inject_defects.py` | ✅ Complete |
| Dataset versioning (DVC) | M2/M3 | `dvc.yaml` | ⬜ Sprint 1 |
| Ingestion + cleaning | M2 | `src/ingest/` | ⬜ Sprint 1 |
| Shared feature module | M2 | `src/features/` | ⬜ Sprint 1 |
| Offline feature store | M2 | `data/processed/` | ⬜ Sprint 1 |
| MLflow experiment tracking | M3 | `src/training/` | ⬜ Sprint 2 |
| Model registry + selection | M3 | MLflow registry | ⬜ Sprint 2 |
| Reproducibility audit | M3 | `reports/` | ⬜ Sprint 2 |
| FastAPI inference service | M4 | `src/serving/` | ⬜ Sprint 3 |
| Docker packaging | M4 | `Dockerfile` | ⬜ Sprint 3 |
| API contract tests | M4 | `tests/` | ⬜ Sprint 3 |
| Prediction logging | M5 | `src/monitoring/` | ⬜ Sprint 4 |
| Drift detection | M5 | `src/monitoring/` | ⬜ Sprint 4 |
| Retraining trigger design | M5 | `docs/` | ⬜ Sprint 4 |
| Governance record | M6 | this file | 🔄 In progress |

---

## Data Quality Findings

Ingestion-boundary validation rejected one candidate corpus and identified three
defects in the accepted corpus. Full evidence in
[`docs/DATA_QUALITY_INCIDENT.md`](docs/DATA_QUALITY_INCIDENT.md).

| ID | Finding | Disposition |
|---|---|---|
| — | Sentiment140 export truncated at 2²⁰ rows; 76/24 class skew is a corruption artefact | Corpus **rejected** |
| BR-01 | `Country` repeats on a fixed 195-row cycle — positionally appended, not key-joined | 6 columns **excluded** |
| BR-02 | `selected_text` is a span of the target label | **Excluded** (leakage) |
| BR-03 | 39.9% of texts carry leading whitespace | Normalised in **shared** module |
| BR-04 | 1 null text (`textID=fdb77c3752`) | Dropped, **logged** |

---

## Model Selection Decision

> To be completed in Sprint 2. Will record: selected MLflow Run ID, exact
> parameters, metrics, **why this run**, and **why not** each rejected run —
> per the M3 §3.3 selection discipline.

---

## Governance Checklist

```
[ ✅ ] Raw data schema validated before any downstream stage
[ ✅ ] Validation fails loudly (exit 1) rather than warning and continuing
[ ✅ ] Validation gate proven to fail via injected-defect negative test
[ ✅ ] Data quality findings documented with evidence, not just fixed
[ ✅ ] Fabricated / leaking columns excluded with recorded justification
[ ✅ ] Environment pinned via conda environment.yml
[ ⬜ ] Dataset versioned and tagged with DVC
[ ⬜ ] Feature logic centralised — not duplicated in training and serving
[ ⬜ ] All experiments logged in MLflow with full parameters
[ ⬜ ] Winning run reproducible from logged configuration
[ ⬜ ] Inference API has Pydantic input validation
[ ⬜ ] Model artifact frozen and immutable after selection
[ ⬜ ] All predictions logged with inputs, outputs, model version
[ ⬜ ] Drift detection produces actionable output
[ ⬜ ] Retraining trigger designed with justified thresholds
[ ⬜ ] Docker containerisation
[ ⬜ ] Unit tests on the feature pipeline
```

---

## Residual Risks

- **Label accuracy is unverifiable.** M2 §2.2.2 — no automated check exists for
  label correctness. A sampled label audit is out of scope for a 14-day project.
- **`test.csv` understates temporal drift.** Drawn from the same collection
  period as the training split, so it is a conservative production simulator.
- **SQLite feature store is not production-grade** under concurrency. Chosen
  deliberately for scope; the limitation is architectural, not accidental.

---

## Owner & Accountability

```
System owner      : Narayan Prasad (2025paml567)
Retraining owner  : same — solo project
Last updated      : Sprint 0
```

---

## References

- **T1** Crowe, R. et al. *Machine Learning Production Systems.* O'Reilly, 2024.
- **T2** Burkov, A. *Machine Learning Engineering.* 2020.
- **R1** McMahon, A.P. *Machine Learning Engineering with Python*, 2nd ed. Packt, 2023.
- Dataset: Tweet Sentiment Extraction (Kaggle). Sentiment140 (Go, Bhayani & Huang, 2009) — evaluated and rejected; see incident report.
