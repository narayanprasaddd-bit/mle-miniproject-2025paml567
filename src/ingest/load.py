"""
Data ingestion and cleaning, with a full audit trail.

Usage:
    python -m src.ingest.load --config configs/sentiment.yaml

Design principle (M2 2.2): every row that leaves this stage differently from how
it arrived must be ACCOUNTED FOR. A row count that changes without an audit
record is indistinguishable from a pipeline bug.

The audit record is written to reports/ingest_audit.json and is the artefact
that answers "why does the feature store have 27,480 rows when the source file
has 27,481?" without anyone having to read the code.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

AUDIT_PATH = Path("reports/ingest_audit.json")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


class IngestAudit:
    """Accumulates a record of every transformation applied to the data."""

    def __init__(self, source: str, initial_rows: int) -> None:
        self.source = source
        self.initial_rows = initial_rows
        self.steps: list[dict] = []

    def record(self, action: str, rows_before: int, rows_after: int, reason: str) -> None:
        self.steps.append(
            {
                "action": action,
                "rows_before": rows_before,
                "rows_after": rows_after,
                "rows_affected": rows_before - rows_after,
                "reason": reason,
            }
        )

    def to_dict(self, final_rows: int) -> dict:
        return {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "source": self.source,
            "initial_rows": self.initial_rows,
            "final_rows": final_rows,
            "total_rows_dropped": self.initial_rows - final_rows,
            "steps": self.steps,
        }


def clean(df: pd.DataFrame, cfg: dict, audit: IngestAudit) -> pd.DataFrame:
    """
    Apply cleaning steps in a fixed, logged order.

    Order matters and is not arbitrary:
      1. drop nulls in the text column  -- cannot featurise a null
      2. strip whitespace               -- BR-03 skew vector
      3. drop rows that became empty    -- whitespace-only text is not text
      4. drop excluded columns          -- leakage and fabricated metadata
    """
    data_cfg = cfg["data"]
    text_col = data_cfg["text_column"]

    # ---- 1. null text -------------------------------------------------
    before = len(df)
    df = df.dropna(subset=[text_col])
    if before != len(df):
        audit.record(
            "drop_null_text",
            before,
            len(df),
            f"BR-04: '{text_col}' was null; a null cannot be featurised",
        )

    # ---- 2. normalise whitespace (BR-03) ------------------------------
    # Applied here AND in the shared feature module. Belt and braces: the
    # feature module is the contract, this is defence in depth for the
    # persisted feature store.
    n_leading = int(df[text_col].astype(str).str.startswith(" ").sum())
    df = df.copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    audit.record(
        "strip_whitespace",
        len(df),
        len(df),
        f"BR-03: {n_leading} rows had leading whitespace (skew vector); "
        f"stripped identically to the serving path",
    )

    # ---- 3. rows that are now empty -----------------------------------
    before = len(df)
    df = df[df[text_col].str.len() > 0]
    if before != len(df):
        audit.record(
            "drop_empty_after_strip",
            before,
            len(df),
            "text was whitespace-only; no tokens can be extracted",
        )

    # ---- 4. drop excluded columns -------------------------------------
    excluded = [c for c in data_cfg.get("excluded_columns", []) if c in df.columns]
    if excluded:
        df = df.drop(columns=excluded)
        audit.record(
            "drop_excluded_columns",
            len(df),
            len(df),
            f"dropped {len(excluded)} column(s) per config: {', '.join(excluded)} "
            f"(leakage + fabricated metadata, see DATA_QUALITY_INCIDENT.md)",
        )

    return df.reset_index(drop=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Ingest and clean raw data")
    ap.add_argument("--config", required=True)
    ap.add_argument("--input", default=None, help="Override config data.raw_train")
    ap.add_argument(
        "--output",
        default="data/processed/clean.parquet",
        help="Where to write the cleaned frame",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    src = args.input or cfg["data"]["raw_train"]

    print("=" * 72)
    print(f"  INGESTION  |  task={cfg['task']['name']}  |  {src}")
    print("=" * 72)

    if not Path(src).exists():
        print(f"  FAIL: input not found: {src}")
        return 1

    df = pd.read_csv(src, encoding=cfg["data"]["encoding"])
    audit = IngestAudit(src, len(df))
    print(f"  loaded {len(df):,} rows x {len(df.columns)} columns")

    df = clean(df, cfg, audit)

    print(f"  cleaned to {len(df):,} rows x {len(df.columns)} columns\n")
    print("  audit trail:")
    for step in audit.steps:
        affected = step["rows_affected"]
        marker = f"-{affected} rows" if affected else "in place"
        print(f"    [{marker:>12}]  {step['action']}")
        print(f"                    {step['reason']}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)

    AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_PATH, "w", encoding="utf-8") as fh:
        json.dump(audit.to_dict(len(df)), fh, indent=2)

    print(f"\n  wrote {out}")
    print(f"  audit: {AUDIT_PATH}")
    print(
        f"\n  RESULT: {audit.initial_rows:,} -> {len(df):,} rows "
        f"({audit.initial_rows - len(df)} dropped, all accounted for)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
