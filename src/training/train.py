"""
MLflow-tracked training experiments.

Usage:
    python -m src.training.train --config configs/sentiment.yaml --run baseline
    python -m src.training.train --config configs/sentiment.yaml --run all

--------------------------------------------------------------------------
EXPERIMENT DISCIPLINE  (M3 3.3)
--------------------------------------------------------------------------
Three rules govern every run below, and each is a rubric line:

  1. BASELINE FIRST.  Run 1 is the simplest thing that could work. Without it,
     no later number means anything. "F1 = 0.71" is unanswerable until you know
     what a linear model on word counts achieves.

  2. ONE VARIABLE PER RUN.  Each run changes exactly ONE thing from the
     baseline. If two things change and the score moves, the experiment has
     told you nothing about which one caused it.

  3. HYPOTHESIS BEFORE RUN.  Each run below states, in its docstring, what it
     expects and why. Recording the prediction before seeing the result is what
     separates an experiment from a fishing expedition.

Every run logs: parameters, metrics, the fitted model, the feature bundle, the
data version, and the git commit. That set is the reproducibility contract --
enough to rebuild the run from scratch months later.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from src.features.text_features import load_bundle

STORE_PATH = Path("data/processed/feature_store.db")
BUNDLE_PATH = Path("models/feature_bundle.joblib")
COMPARISON_PATH = Path("reports/model_comparison.json")
EXPERIMENT_NAME = "sentiment-classification"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def git_commit() -> str:
    """Record the exact code state. Part of the reproducibility contract."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def load_splits(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the persisted feature store. Features are never recomputed here."""
    with sqlite3.connect(STORE_PATH) as conn:
        train = pd.read_sql("SELECT * FROM features_train", conn)
        test = pd.read_sql("SELECT * FROM features_test", conn)
    return train, test


# ---------------------------------------------------------------------------
# Run definitions -- each states its hypothesis before it runs
# ---------------------------------------------------------------------------

RUNS = {
    "baseline": {
        "name": "run1_baseline_logreg",
        "hypothesis": (
            "A linear model on TF-IDF features establishes the floor. Text "
            "classification on lexically obvious sentiment is largely a "
            "word-presence problem, so this should already be competitive."
        ),
        "varied": "none (baseline)",
        "model": lambda cfg: LogisticRegression(
            max_iter=1000, random_state=cfg["training"]["random_state"]
        ),
        "params": {"model_family": "LogisticRegression", "C": 1.0, "max_iter": 1000},
    },
    "svc": {
        "name": "run2_linearsvc",
        "hypothesis": (
            "LinearSVC optimises a hinge loss and a maximum margin rather than "
            "likelihood. On high-dimensional sparse text it often edges out "
            "logistic regression. Expected: comparable, possibly +0.01 F1. "
            "Trade-off: no native predict_proba, which the serving layer and "
            "the M5 confidence drift signal both require."
        ),
        "varied": "algorithm (logistic -> hinge loss)",
        "model": lambda cfg: LinearSVC(random_state=cfg["training"]["random_state"]),
        "params": {"model_family": "LinearSVC", "C": 1.0},
    },
    "regularised": {
        "name": "run3_logreg_C0.5",
        "hypothesis": (
            "20,000 features against 21,984 training rows is a wide problem, so "
            "stronger regularisation may generalise better. Halving C doubles "
            "the penalty. Expected: small change either way -- which is itself "
            "the useful result, because a flat response means the baseline was "
            "not overfitting."
        ),
        "varied": "regularisation strength (C: 1.0 -> 0.5)",
        "model": lambda cfg: LogisticRegression(
            C=0.5, max_iter=1000, random_state=cfg["training"]["random_state"]
        ),
        "params": {"model_family": "LogisticRegression", "C": 0.5, "max_iter": 1000},
    },
}


def evaluate(model, X_test, y_test) -> dict:
    """Metrics chosen for the task, not by convention."""
    pred = model.predict(X_test)
    return {
        # f1_macro is the primary metric: 3 classes with mild imbalance
        # (neutral 40%, positive 31%, negative 28%). Macro weights each class
        # equally, so the model cannot win by neglecting the smallest class.
        "f1_macro": f1_score(y_test, pred, average="macro"),
        "f1_weighted": f1_score(y_test, pred, average="weighted"),
        "accuracy": accuracy_score(y_test, pred),
        "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
    }


def run_experiment(key: str, cfg: dict, train, test, bundle) -> dict:
    spec = RUNS[key]
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    X_train = bundle.transform(train[text_col].tolist())
    X_test = bundle.transform(test[text_col].tolist())
    y_train, y_test = train[label_col], test[label_col]

    with mlflow.start_run(run_name=spec["name"]):
        model = spec["model"](cfg)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)

        # ---- parameters ------------------------------------------------
        mlflow.log_params(spec["params"])
        mlflow.log_params(
            {
                "varied_from_baseline": spec["varied"],
                "random_state": cfg["training"]["random_state"],
                "n_features": bundle.n_features,
                "vocabulary_size": bundle.vocabulary_size,
                "n_train": len(train),
                "n_test": len(test),
                **{f"vec_{k}": v for k, v in bundle.config_fingerprint.items()},
            }
        )

        # ---- reproducibility contract ----------------------------------
        mlflow.set_tags(
            {
                "hypothesis": spec["hypothesis"],
                "git_commit": git_commit(),
                "data_version_tag": "v1.0-raw",
                "feature_bundle": str(BUNDLE_PATH),
            }
        )

        mlflow.log_metrics(metrics)

        # The model AND the vectorizer are logged together. Either alone is
        # meaningless to the other -- see src/features/text_features.py.
        mlflow.sklearn.log_model(model, name="model")
        mlflow.log_artifact(str(BUNDLE_PATH), artifact_path="feature_bundle")

        report = classification_report(
            y_test, model.predict(X_test), zero_division=0
        )
        Path("reports").mkdir(exist_ok=True)
        rp = Path(f"reports/classification_report_{spec['name']}.txt")
        rp.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(rp))

        run_id = mlflow.active_run().info.run_id

    print(f"  {spec['name']:26s} f1_macro={metrics['f1_macro']:.4f}  "
          f"acc={metrics['accuracy']:.4f}  run_id={run_id[:8]}")

    return {
        "run": spec["name"],
        "run_id": run_id,
        "varied": spec["varied"],
        "hypothesis": spec["hypothesis"],
        "params": spec["params"],
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="MLflow-tracked training runs")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--run",
        default="all",
        choices=list(RUNS.keys()) + ["all"],
        help="Which experiment to run",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    if not STORE_PATH.exists() or not BUNDLE_PATH.exists():
        print("  FAIL: feature store or bundle missing. Run `dvc repro` first.")
        return 1

    mlflow.set_experiment(EXPERIMENT_NAME)
    train, test = load_splits(cfg)
    bundle = load_bundle(BUNDLE_PATH)

    print("=" * 72)
    print(f"  TRAINING  |  experiment={EXPERIMENT_NAME}")
    print("=" * 72)
    print(f"  train={len(train):,}  test={len(test):,}  "
          f"features={bundle.n_features:,}  seed={cfg['training']['random_state']}")
    print(f"  git={git_commit()}  data=v1.0-raw\n")

    keys = list(RUNS.keys()) if args.run == "all" else [args.run]
    results = [run_experiment(k, cfg, train, test, bundle) for k in keys]

    if len(results) > 1:
        best = max(results, key=lambda r: r["metrics"]["f1_macro"])
        payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "primary_metric": cfg["training"]["primary_metric"],
            "runs": results,
            "selected_run": best["run"],
            "selected_run_id": best["run_id"],
        }
        COMPARISON_PATH.parent.mkdir(exist_ok=True)
        with open(COMPARISON_PATH, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\n  best by f1_macro: {best['run']} ({best['metrics']['f1_macro']:.4f})")
        print(f"  comparison written: {COMPARISON_PATH}")
        print("\n  NOTE: highest score is a CANDIDATE, not a decision.")
        print("  See docs/MODEL_SELECTION.md for the selection rationale.")

    print("\n  view runs with:  mlflow ui")
    return 0


if __name__ == "__main__":
    sys.exit(main())
