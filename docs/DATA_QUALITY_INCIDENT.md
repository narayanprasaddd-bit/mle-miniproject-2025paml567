# Data Quality Incident Report

**Project:** Mini-Project I — Flavor C, Text Classification
**Course:** Machine Learning Engineering (PCAM ZC412)
**Module reference:** M2 — Data Engineering for Machine Learning
**Date raised:** 10 August 2026
**Status:** Closed — candidate corpus rejected, defective columns excluded

---

## 1. Summary

The dataset archive obtained for this project contained **four CSV files drawn
from two unrelated Kaggle datasets**. Ingestion-boundary validation rejected one
corpus outright and identified three defects in the corpus that was accepted.

| File | Origin dataset | Rows | Decision |
|---|---|---|---|
| `train.csv` | Tweet Sentiment Extraction | 27,481 | **Accepted** — primary training corpus |
| `test.csv` | Tweet Sentiment Extraction | 4,815 | **Accepted** — held-out production simulator |
| `training.1600000.processed.noemoticon.csv` | Sentiment140 | 1,048,572 | **Rejected** — see §2 |
| `testdata.manual.2009.06.14.csv` | Sentiment140 | 516 | Not used — orphaned by the above |

All four findings were surfaced by automated validation **before** any feature
engineering or model training ran. This is the cost asymmetry described in
M2 §2.5: a defect caught at the ingestion boundary is cheap; the same defect
discovered after deployment requires retraining, rollback, and an audit of past
predictions.

---

## 2. Finding 1 — Sentiment140 export is truncated (BLOCKING, corpus rejected)

### Evidence

Three independent signals, all consistent with the same root cause:

**2.1 Row count sits at the Excel worksheet ceiling.**
The file contains 1,048,572 data rows. Adding the header gives 1,048,573.
Microsoft Excel's maximum worksheet size is 1,048,576 rows (2^20). The
published Sentiment140 corpus contains 1,600,000 rows.

**2.2 Class balance is destroyed.**
Sentiment140 is distributed with exactly 800,000 negative and 800,000 positive
records, stored **sorted by label** — all negatives first. The observed
distribution:

| Polarity | Expected | Observed | Delta |
|---|---|---|---|
| 0 (negative) | 800,000 | 799,996 | −4 |
| 4 (positive) | 800,000 | 248,576 | **−551,424** |

The negative block survived almost intact; the positive block lost 69% of its
records. That is the exact signature of a truncation at a fixed row offset
against label-sorted data. The resulting 76/24 split is an **artefact of the
corruption, not a property of the data.**

**2.3 Header artefacts confirm manual editing.**
The canonical Sentiment140 file has **no header row**. This file has one, and
its column names contain U+00A0 non-breaking spaces
(`'polarity of tweet\xa0'`) — a characteristic spreadsheet export artefact.

### Root cause

The file was opened in a spreadsheet application, silently truncated at the
worksheet row limit, and re-saved. No error was raised at any point.

### Why this matters beyond this project

This is precisely the failure class M2 §2.1 opens with. The file loads without
error. `pandas.read_csv` returns a valid DataFrame. Schema validation passes —
every column has the right type and every value is in range. A model trained on
it would converge, report plausible metrics, and be deployed.

The defect is only visible through **Level 3 statistical validation**: comparing
the observed class distribution against a documented expected baseline. Without
that check, this corpus would have produced a model with a 76/24 prior baked in,
learned from a dataset that is genuinely balanced.

### Action taken

Corpus rejected. The file is retained in the repository history (via DVC) as a
documented negative example rather than deleted, so the finding is auditable.

---

## 3. Finding 2 — `Country` and derived columns are fabricated (BR-01)

### Evidence

The `Country` column cycles alphabetically through 195 country names and repeats
on a fixed period:

```
row   0  Afghanistan          row 195  Afghanistan
row   1  Albania              row 196  Albania
row   2  Algeria              row 197  Algeria
row 193  Zambia
row 194  Zimbabwe
```

Automated check: `column[i] == column[i + 195]` held for **585 of 585** compared
positions. 27,481 rows ÷ 195 = 140.9 complete cycles.

A genuine per-record attribute cannot repeat on a fixed row cycle. The column
was **appended by row position, not joined by key**, and therefore carries no
information about the tweet beside it. `Population -2020`, `Land Area (Km²)` and
`Density (P/Km²)` are lookups derived from it, so they inherit the same defect.

### Corroboration — the metadata carries no signal

`Time of Tweet` and `Age of User` were tested for association with the target:

| Time of Tweet | negative | neutral | positive |
|---|---|---|---|
| morning | 27.96% | 41.08% | 30.97% |
| night | 28.58% | 40.17% | 31.24% |
| noon | 28.41% | 40.12% | 31.47% |

Flat to within 0.5 percentage points across all three strata. No usable signal.

### Action taken

All six metadata columns excluded from the feature set via
`configs/sentiment.yaml → data.excluded_columns`. Business rule **BR-01** was
added to `src/validation/schema.py` so the detection is automated and repeatable
rather than a one-off observation.

**Engineering note.** These columns are the more dangerous kind of defect: not
missing data, but *noise presented as a feature*. Included in a TF-IDF pipeline
alongside a one-hot country encoding, they would add 195 dimensions of pure
noise, inflate variance, and — with a sufficiently flexible model — invite
overfitting to a cycle that is an artefact of row order.

---

## 4. Finding 3 — Leading whitespace on 39.9% of records (BR-03)

10,974 of 27,480 non-null `text` values begin with a space.

Harmless in itself. Dangerous as a **training–serving skew vector** (M2 §2.6):
if training strips leading whitespace and the serving path does not, the two
paths tokenise the same input differently and every prediction shifts. The
failure is silent — the API returns HTTP 200 throughout.

**Action taken.** Whitespace normalisation lives in the shared feature module
(`src/features/`), imported by both the training script and the serving
application. Neither path implements its own normalisation. Business rule BR-03
records the evidence that justifies the step.

---

## 5. Finding 4 — One null text value (BR-04)

`textID = fdb77c3752` has a null `text` and a null `selected_text`, with
`sentiment = neutral`.

**Action taken.** Dropped during cleaning, with the drop **counted and logged**.
Silent filtering is prohibited: a row count that changes without an audit record
is indistinguishable from a pipeline bug.

---

## 6. Leakage check — `selected_text`

`selected_text` is the annotated span of `text` that carries the sentiment. It
is a **derivative of the target label** and is not available at inference time.

Excluded from features. Retained in the raw data for the M6 explainability
discussion, where the annotated spans provide a ground-truth reference for
comparing against model feature-importance output.

---

## 7. Validation coverage summary

| Level | Mechanism | Findings raised here |
|---|---|---|
| L1 Schema | Pandera `DataFrameModel`, `coerce=False` | — (passed) |
| L2 Range/domain | Pandera `Field(isin=...)`, length bounds | — (passed) |
| L3 Statistical | Label-proportion baseline, uniqueness | **Finding 1** |
| L4 Business rule | BR-01 … BR-04 | **Findings 2, 3, 4** |

Reproduce with:

```bash
python -m src.validation.validate --config configs/sentiment.yaml
```

Negative test — confirm the pipeline halts on injected defects:

```bash
python -m tests.inject_defects          # writes a corrupted copy
python -m src.validation.validate --config configs/sentiment.yaml \
    --input data/raw/_corrupt_test.csv  # expect exit code 1
```

---

## 8. Residual risk

- **Label accuracy is not verifiable.** M2 §2.2.2 notes there is no automated
  check for label correctness. The sentiment annotations are taken on trust; a
  label audit on a sampled subset is out of scope for a 14-day project and is
  recorded here as a known limitation.
- **`test.csv` is used as a production simulator.** It shares 0 of 4,815
  `textID` values and only 1 of 4,815 raw text strings with `train.csv`, so
  leakage is negligible — but it is drawn from the same collection period and
  therefore understates real-world temporal drift.
