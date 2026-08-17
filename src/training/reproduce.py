"""
Reproducibility audit.

Usage:
    python -m src.training.reproduce --config configs/sentiment.yaml --run-id <id>

--------------------------------------------------------------------------
THE CLAIM BEING TESTED
--------------------------------------------------------------------------
"This run is reproducible" is an assertion until someone tries it. This script
tries it.

It reads a completed MLflow run, extracts ONLY the logged parameters -- not the
in-memory objects, not the original script's local variables -- rebuilds the
model from those parameters alone, retrains, and compares the resulting metrics
against what was originally logged.

Agreement to four decimal places means the reproducibility contract holds: the
logged configuration is sufficient to rebuild the run. Disagreement means
something influenced the result that was never recorded -- an unseeded random
draw, a library version, an environment variable, a data version -- and the
run's provenance is broken.

M3 3.2 names five sources of irreproducibility:
    1. random seeds          -> logged as a parameter, asserted below
    2. environment drift     -> pinned in environment.yml
    3. data version drift    -> logged as the data_version_tag
    4. code state            -> logged as the git commit
    5. hardware/parallelism  -> not applicable: single-threaded CPU, no GPU
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

import mlflow
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

from src.features.text_features import load_bundle

STORE_PATH = Path("data/processed/feature_store.db")
BUNDLE_PATH = Path("models/feature_bundle.joblib")
OUT_PATH = Path("reports/reproducibility_audit.json")

TOLERANCE = 1e-4


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def rebuild_model(params: dict):
    """
    Reconstruct the estimator from LOGGED PARAMETERS ONLY.

    This function deliberately has no access to the training script's objects.
    If a parameter was not logged, it cannot be recovered here -- which is
    exactly the property the audit is testing.
    """
    family = params["model_family"]
    seed = int(params["random_state"])

    if family == "LogisticRegression":
        return LogisticRegression(
            C=float(params["C"]),
            max_iter=int(params["max_iter"]),
            random_state=seed,
        )
    if family == "LinearSVC":
        return LinearSVC(C=float(params["C"]), random_state=seed)

    raise ValueError(f"Unknown model_family in logged params: {family}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproducibility audit")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--run-id",
        default=None,
        help="MLflow run id. Defaults to the selected run in model_comparison.json",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]

    run_id = args.run_id
    if run_id is None:
        comparison = Path("reports/model_comparison.json")
        if not comparison.exists():
            print("  FAIL: no --run-id given and reports/model_comparison.json missing.")
            return 1
        run_id = json.loads(comparison.read_text())["selected_run_id"]

    print("=" * 72)
    print(f"  REPRODUCIBILITY AUDIT  |  run_id={run_id}")
    print("=" * 72)

    client = mlflow.tracking.MlflowClient()
    run = client.get_run(run_id)
    logged_params = run.data.params
    logged_metrics = run.data.metrics

    print(f"  run name : {run.info.run_name}")
    print(f"  git      : {run.data.tags.get('git_commit', 'not logged')}")
    print(f"  data     : {run.data.tags.get('data_version_tag', 'not logged')}")
    print("\n  logged parameters used for the rebuild:")
    for key in ("model_family", "C", "max_iter", "random_state"):
        if key in logged_params:
            print(f"    {key:16s} = {logged_params[key]}")

    # ---- rebuild and retrain from logged params only --------------------
    with sqlite3.connect(STORE_PATH) as conn:
        train = pd.read_sql("SELECT * FROM features_train", conn)
        test = pd.read_sql("SELECT * FROM features_test", conn)

    bundle = load_bundle(BUNDLE_PATH)
    X_train = bundle.transform(train[text_col].tolist())
    X_test = bundle.transform(test[text_col].tolist())

    model = rebuild_model(logged_params)
    model.fit(X_train, train[label_col])
    pred = model.predict(X_test)

    recomputed = {
        "f1_macro": f1_score(test[label_col], pred, average="macro"),
        "accuracy": accuracy_score(test[label_col], pred),
    }

    # ---- compare --------------------------------------------------------
    print("\n  metric comparison:")
    print(f"    {'metric':16s} {'logged':>10s} {'recomputed':>12s} {'delta':>12s}")
    all_match = True
    comparison_rows = {}
    for name, value in recomputed.items():
        original = logged_metrics.get(name)
        if original is None:
            continue
        delta = abs(original - value)
        # bool() is not cosmetic: original - value is a numpy float, so the
        # comparison yields numpy.bool_, which json.dump cannot serialise.
        # This raised "TypeError: Object of type bool_ is not JSON serializable"
        # and failed the dvc `reproduce` stage after the comparison had already
        # printed correctly -- a reminder that a passing console output is not
        # the same as a passing stage.
        match = bool(delta < TOLERANCE)
        all_match = bool(all_match and match)
        comparison_rows[name] = {
            "logged": round(original, 6),
            "recomputed": round(value, 6),
            "delta": round(delta, 8),
            "match": match,
        }
        flag = "OK" if match else "MISMATCH"
        print(f"    {name:16s} {original:10.4f} {value:12.4f} {delta:12.2e}  {flag}")

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "git_commit": run.data.tags.get("git_commit"),
                "data_version_tag": run.data.tags.get("data_version_tag"),
                "tolerance": TOLERANCE,
                "metrics": comparison_rows,
                "reproducible": all_match,
            },
            fh,
            indent=2,
        )

    print(f"\n  written: {OUT_PATH}")
    if all_match:
        print("\n  RESULT: REPRODUCIBLE -- logged configuration is sufficient")
        print("  to rebuild this run from scratch.")
        return 0

    print("\n  RESULT: NOT REPRODUCIBLE -- something influenced the original")
    print("  result that was never logged. Provenance is broken.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
