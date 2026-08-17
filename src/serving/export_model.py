"""
Export the registered model from MLflow to a plain joblib file for serving.

Usage:
    python -m src.serving.export_model

WHY NOT LOAD FROM MLFLOW AT RUNTIME?

The serving container should not depend on the MLflow tracking server. If the
tracking server is down, or the network is partitioned, or the run is later
archived, inference must keep working. So the selected model is exported once,
at build time, into an immutable artefact that the container carries with it.

This is the M4 "frozen artefact" principle: after selection the model is
immutable. A new model means a new artefact and a new deployment, never an
in-place mutation of a running service.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import mlflow

MODEL_PATH = Path("models/classifier.joblib")
COMPARISON = Path("reports/model_comparison.json")


def main() -> int:
    if not COMPARISON.exists():
        print("  FAIL: run training first.")
        return 1

    payload = json.loads(COMPARISON.read_text())
    run_id = payload["selected_run_id"]
    run_name = payload["selected_run"]

    print("=" * 72)
    print(f"  EXPORT MODEL  |  {run_name}  |  run {run_id[:8]}")
    print("=" * 72)

    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    sidecar = MODEL_PATH.with_suffix(".json")
    sidecar.write_text(
        json.dumps(
            {
                "source_run_id": run_id,
                "source_run_name": run_name,
                "classes": [str(c) for c in model.classes_],
                "note": (
                    "Frozen at export. Must be deployed together with "
                    "models/feature_bundle.joblib - either alone is invalid."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"  wrote {MODEL_PATH}")
    print(f"  classes: {list(model.classes_)}")
    print("\n  Deploy this ALONGSIDE models/feature_bundle.joblib.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
