# Sprint 2 — Paste-Ready Commands

**Sprint 2 = M3 Experimentation & Reproducibility → 20% of the grade.**

Anaconda Prompt, `(mle)` active, from the repo root.

---

## 1. Unpack

Git Bash:

```bash
cd /c/projects/mle-miniproject-2025paml567
unzip -o "/c/NP Documents and Tools/BITS/Mini Project/sprint2_files.zip" -d .
```

---

## 2. Install MLflow (if not already present)

```
pip install mlflow
mlflow --version
```

---

## 3. Run the three experiments

```
python -m src.training.train --config configs/sentiment.yaml --run all
```

Expected (your numbers may differ in the 4th decimal):

```
run1_baseline_logreg       f1_macro=0.6808  acc=0.6794
run2_linearsvc             f1_macro=0.6716  acc=0.6685
run3_logreg_C0.5           f1_macro=0.6789  acc=0.6790
best by f1_macro: run1_baseline_logreg
```

---

## 4. Test whether the differences are real

```
python -m src.training.significance --config configs/sentiment.yaml
```

Expected: run2 difference **MEANINGFUL**, run3 difference **NOISE**.

---

## 5. Reproducibility audit

```
python -m src.training.reproduce --config configs/sentiment.yaml
```

Expected: `RESULT: REPRODUCIBLE` with delta 0.00e+00 on both metrics.

---

## 6. Register the model

```
python -m src.training.register --config configs/sentiment.yaml
```

---

## 7. Screenshot the MLflow UI  ← REQUIRED DELIVERABLE

The brief asks for "experiment tracking logs and a short model comparison
report (screenshots or exported logs, e.g., MLflow)."

```
mlflow ui
```

Open http://127.0.0.1:5000 and capture:

1. The **runs table** showing all three runs with their metrics
2. One **run detail** page showing parameters, metrics, and tags
3. The **Models** tab showing `sentiment-classifier` version 1 in Staging

Save them into `docs/screenshots/`. Press Ctrl+C in the terminal to stop the UI.

---

## 8. Commit

Git Bash:

```bash
git add -A
git status --short
git commit -m "Sprint 2: MLflow experiments, significance testing, model selection" -m "- Three tracked runs, one variable changed per run, hypothesis recorded before each" -m "- Baseline logistic regression selected; recorded as a legitimate outcome" -m "- Bootstrap paired test: run2 difference real, run3 difference indistinguishable from noise" -m "- Reproducibility audit: run rebuilt from logged params alone, metrics match exactly" -m "- LinearSVC rejected on two grounds: lower score and no predict_proba for the M5 drift signal" -m "- Model registered to MLflow registry, Staging, with approver and gated promotion"
git push
```
