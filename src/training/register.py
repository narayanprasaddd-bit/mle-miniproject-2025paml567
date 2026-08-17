"""
Register the selected model in the MLflow Model Registry.

Usage:
    python -m src.training.register --config configs/sentiment.yaml

M3 3.4: the registry is the record of WHICH model is live, WHO approved it, and
WHY. Lifecycle: None -> Staging -> Production -> Archived.

Promotion to Production is deliberately NOT automatic here. A model reaches
Staging on merit; it reaches Production only after the M4 contract tests pass
and the M5 monitoring baseline exists. That gate is a governance decision, not
a metric threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

MODEL_NAME = "sentiment-classifier"
APPROVER = "Narayan Prasad (2025paml567)"


def main() -> int:
    ap = argparse.ArgumentParser(description="Register the selected model")
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args()

    run_id = args.run_id
    if run_id is None:
        comparison = Path("reports/model_comparison.json")
        if not comparison.exists():
            print("  FAIL: run training first.")
            return 1
        run_id = json.loads(comparison.read_text())["selected_run_id"]

    print("=" * 72)
    print(f"  MODEL REGISTRY  |  {MODEL_NAME}")
    print("=" * 72)

    client = MlflowClient()
    uri = f"runs:/{run_id}/model"

    result = mlflow.register_model(uri, MODEL_NAME)
    version = result.version
    print(f"  registered version {version} from run {run_id[:8]}")

    client.set_model_version_tag(MODEL_NAME, version, "approver", APPROVER)
    client.set_model_version_tag(
        MODEL_NAME, version, "selection_record", "docs/MODEL_SELECTION.md"
    )
    client.set_model_version_tag(MODEL_NAME, version, "data_version", "v1.0-raw")
    client.set_model_version_tag(
        MODEL_NAME,
        version,
        "promotion_gate",
        "Production requires: M4 contract tests pass + M5 baseline established",
    )

    client.set_registered_model_alias(MODEL_NAME, "staging", version)
    print(f"  alias 'staging' -> version {version}")
    print(f"  approver: {APPROVER}")
    print("\n  Production promotion is gated, not automatic. See docs/MODEL_SELECTION.md.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
