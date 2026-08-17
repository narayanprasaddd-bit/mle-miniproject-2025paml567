"""
Build the offline feature store.

Usage:
    python -m src.features.build --config configs/sentiment.yaml

Responsibilities, in order:
    1. split train/test FIRST                 -- before any fitting
    2. fit the vectorizer on the TRAIN split only
    3. persist the bundle (the contract)
    4. write both splits to a SQLite feature store
    5. record a manifest for reproducibility

--------------------------------------------------------------------------
WHY SPLIT BEFORE FIT
--------------------------------------------------------------------------
If the vectorizer is fitted on all the data, its vocabulary and IDF weights
carry information from the test set. The test score is then optimistic, because
the model was evaluated on text whose token statistics it had already seen.

This is the same rule as fitting StandardScaler on training data only. In the
Core ML coursework it was called data leakage; here it is the same defect
viewed from the serving side, where the "test set" is production traffic the
vectorizer definitionally cannot have seen.

The ordering below is therefore load -> split -> fit, never load -> fit -> split.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

from src.features.text_features import fit_vectorizer, save_bundle

STORE_PATH = Path("data/processed/feature_store.db")
BUNDLE_PATH = Path("models/feature_bundle.joblib")
MANIFEST_PATH = Path("reports/feature_manifest.json")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_store(train: pd.DataFrame, test: pd.DataFrame, cfg: dict) -> None:
    """
    Write both splits to SQLite.

    SQLite is chosen deliberately for scope, and the limitation is architectural
    rather than accidental: it does not handle concurrent writers, so it is not
    a production feature store. What it does provide is the property that
    matters pedagogically -- features are computed once, persisted, and read
    identically by every downstream consumer, instead of being recomputed
    inline by each script.
    """
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    id_col = cfg["data"]["id_column"]

    with sqlite3.connect(STORE_PATH) as conn:
        for name, frame in (("train", train), ("test", test)):
            frame[[id_col, text_col, label_col]].to_sql(
                f"features_{name}", conn, if_exists="replace", index=False
            )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_train_id "
            f"ON features_train({id_col})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the offline feature store")
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default="data/processed/clean.parquet")
    args = ap.parse_args()

    cfg = load_config(args.config)
    text_col = cfg["data"]["text_column"]
    label_col = cfg["data"]["label_column"]
    split_cfg = cfg["split"]

    print("=" * 72)
    print(f"  FEATURE BUILD  |  task={cfg['task']['name']}")
    print("=" * 72)

    if not Path(args.input).exists():
        print(f"  FAIL: {args.input} not found. Run src.ingest.load first.")
        return 1

    df = pd.read_parquet(args.input)
    print(f"  loaded {len(df):,} clean rows")

    # ---- 1. SPLIT FIRST ------------------------------------------------
    train, test = train_test_split(
        df,
        test_size=split_cfg["test_size"],
        random_state=split_cfg["random_state"],
        stratify=df[label_col] if split_cfg["stratify"] else None,
    )
    print(
        f"  split  train={len(train):,}  test={len(test):,}  "
        f"(stratified, random_state={split_cfg['random_state']})"
    )

    # ---- 2. FIT ON TRAIN ONLY ------------------------------------------
    bundle = fit_vectorizer(train[text_col].tolist(), cfg["features"])
    print(
        f"  fitted vectorizer on TRAIN ONLY: "
        f"{bundle.vocabulary_size:,} vocabulary terms, "
        f"{bundle.n_features:,} features"
    )

    # ---- 3. PERSIST THE CONTRACT ---------------------------------------
    save_bundle(bundle, BUNDLE_PATH)
    print(f"  wrote contract: {BUNDLE_PATH}")

    # ---- 4. FEATURE STORE ----------------------------------------------
    write_store(train, test, cfg)
    print(f"  wrote feature store: {STORE_PATH}")

    # ---- 5. MANIFEST ---------------------------------------------------
    oov_on_test = bundle.oov_rate(test[text_col].tolist())
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "task": cfg["task"]["name"],
        "source": args.input,
        "rows_total": len(df),
        "rows_train": len(train),
        "rows_test": len(test),
        "split": {
            "test_size": split_cfg["test_size"],
            "random_state": split_cfg["random_state"],
            "stratified": split_cfg["stratify"],
        },
        "vectorizer": bundle.config_fingerprint,
        "vocabulary_size": bundle.vocabulary_size,
        "n_features": bundle.n_features,
        "baseline_oov_rate_on_test": round(oov_on_test, 6),
        "label_distribution_train": train[label_col]
        .value_counts(normalize=True)
        .round(4)
        .to_dict(),
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"  wrote manifest: {MANIFEST_PATH}")
    print(
        f"\n  BASELINE OOV RATE on held-out test = {oov_on_test:.4f}"
        f"\n  ^ this is the M5 drift reference. Production OOV materially above"
        f"\n    this value indicates the language has moved."
    )
    print("\n  RESULT: feature store built.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
