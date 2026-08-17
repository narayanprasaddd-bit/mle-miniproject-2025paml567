"""
Bootstrap significance testing for model comparison.

Usage:
    python -m src.training.significance --config configs/sentiment.yaml

--------------------------------------------------------------------------
WHY THIS EXISTS
--------------------------------------------------------------------------
The instructor's own tutorial poses the question directly when comparing two
runs 0.002 apart:

    "is that difference meaningful, or just noise in the test split?"

Ranking runs by a single test-set number does not answer it. A test set is one
sample; a different 20% split would give different numbers. Declaring a winner
on a 0.002 margin is choosing noise.

This module answers the question empirically. It resamples the test set with
replacement 1,000 times and computes the PAIRED difference in f1_macro between
each candidate and the baseline on every resample. If the resulting 95%
confidence interval excludes zero, the difference survives resampling and is
real. If it spans zero, the two models are indistinguishable on this data and
the choice between them must be made on OTHER grounds -- simplicity,
interpretability, serving cost.

Paired resampling matters: both models are evaluated on the SAME resampled
rows, so per-sample difficulty cancels out and only the model difference
remains.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC

from src.features.text_features import load_bundle

STORE_PATH = Path("data/processed/feature_store.db")
BUNDLE_PATH = Path("models/feature_bundle.joblib")
OUT_PATH = Path("reports/significance_test.json")

N_BOOTSTRAP = 1000
SEED = 42


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap model comparison")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = cfg["training"]["random_state"]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    if not STORE_PATH.exists():
        print("  FAIL: feature store missing. Run `dvc repro` first.")
        return 1

    with sqlite3.connect(STORE_PATH) as conn:
        train = pd.read_sql("SELECT * FROM features_train", conn)
        test = pd.read_sql("SELECT * FROM features_test", conn)

    bundle = load_bundle(BUNDLE_PATH)
    X_train = bundle.transform(train[text_col].tolist())
    X_test = bundle.transform(test[text_col].tolist())
    y_train = train[label_col].values
    y_test = test[label_col].values

    models = {
        "run1_baseline_logreg": LogisticRegression(max_iter=1000, random_state=seed),
        "run2_linearsvc": LinearSVC(random_state=seed),
        "run3_logreg_C0.5": LogisticRegression(C=0.5, max_iter=1000, random_state=seed),
    }

    print("=" * 72)
    print(f"  BOOTSTRAP SIGNIFICANCE  |  B={N_BOOTSTRAP}  |  metric=f1_macro")
    print("=" * 72)

    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)

    # Same resample indices for every model -> paired comparison
    rng = np.random.default_rng(SEED)
    n = len(y_test)
    indices = [rng.integers(0, n, n) for _ in range(N_BOOTSTRAP)]

    scores = {name: [] for name in models}
    for idx in indices:
        y_boot = y_test[idx]
        for name, pred in preds.items():
            scores[name].append(f1_score(y_boot, pred[idx], average="macro"))

    print("\n  95% confidence intervals on test f1_macro:")
    summary = {}
    for name, vals in scores.items():
        arr = np.array(vals)
        lo, hi = np.percentile(arr, [2.5, 97.5])
        summary[name] = {
            "mean": round(float(arr.mean()), 4),
            "ci_lower": round(float(lo), 4),
            "ci_upper": round(float(hi), 4),
        }
        print(f"    {name:24s} {arr.mean():.4f}  [{lo:.4f}, {hi:.4f}]")

    baseline = "run1_baseline_logreg"
    base_arr = np.array(scores[baseline])

    print(f"\n  Paired difference vs {baseline}:")
    comparisons = {}
    for name in models:
        if name == baseline:
            continue
        diff = base_arr - np.array(scores[name])
        lo, hi = np.percentile(diff, [2.5, 97.5])
        meaningful = bool(lo > 0 or hi < 0)
        comparisons[name] = {
            "mean_difference": round(float(diff.mean()), 4),
            "ci_lower": round(float(lo), 4),
            "ci_upper": round(float(hi), 4),
            "meaningful": meaningful,
            "interpretation": (
                "difference survives resampling -- real"
                if meaningful
                else "interval spans zero -- indistinguishable from noise"
            ),
        }
        verdict = "MEANINGFUL" if meaningful else "NOISE"
        print(
            f"    vs {name:22s} {diff.mean():+.4f}  "
            f"[{lo:+.4f}, {hi:+.4f}]  {verdict}"
        )

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "n_bootstrap": N_BOOTSTRAP,
                "seed": SEED,
                "metric": "f1_macro",
                "baseline": baseline,
                "confidence_intervals": summary,
                "paired_comparisons": comparisons,
            },
            fh,
            indent=2,
        )

    print(f"\n  written: {OUT_PATH}")
    print("\n  Where an interval spans zero, the models are NOT distinguishable")
    print("  on this data. Choose on simplicity, interpretability, or serving")
    print("  cost -- and record that reasoning in docs/MODEL_SELECTION.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
