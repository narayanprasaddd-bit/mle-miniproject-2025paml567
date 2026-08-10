"""
Validation pipeline entrypoint.

Usage:
    python -m src.validation.validate --config configs/sentiment.yaml

Exit codes:
    0  all levels passed (findings may still be reported -- see below)
    1  one or more BLOCKING checks failed; downstream stages must not run

The distinction between a FAILURE and a FINDING is deliberate:

    FAILURE  the data cannot be trusted. Stop the pipeline. (Levels 1-3)
    FINDING  the data is usable but a defect was detected and must be handled
             explicitly in configuration. (Level 4)

M2 2.5 framing: "a validation failure is a success, not a system error." The
exit(1) below is the pipeline working correctly.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

try:
    import pandera.pandas as pa
except (ImportError, AttributeError):  # pandera < 0.23
    import pandera as pa

from src.validation.schema import (
    TweetSentimentRawSchema,
    check_fabricated_cycle,
    check_label_distribution,
    check_label_span_consistency,
    check_leading_whitespace,
    check_uniqueness,
)

REPORT_PATH = Path("reports/validation_report.json")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def main() -> int:
    ap = argparse.ArgumentParser(description="Four-level data validation")
    ap.add_argument("--config", required=True, help="Path to a task config YAML")
    ap.add_argument(
        "--input",
        default=None,
        help="Override the input CSV (defaults to config data.raw_train)",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg, schema_cfg = cfg["data"], cfg["schema"]
    src_path = args.input or data_cfg["raw_train"]

    print("=" * 72)
    print(f"  DATA VALIDATION  |  task={cfg['task']['name']}  |  {src_path}")
    print("=" * 72)

    if not Path(src_path).exists():
        print(f"  FAIL [L0] input file not found: {src_path}")
        print("\n  Place train.csv and test.csv in data/raw/ before running.")
        return 1

    df = pd.read_csv(src_path, encoding=data_cfg["encoding"])
    print(f"  loaded {len(df):,} rows x {len(df.columns)} columns\n")

    failures: list[str] = []
    findings: list[str] = []

    # ---------------- LEVEL 1 + 2: schema, dtype, range, domain -------------
    print("  [L1/L2] schema, dtype, and domain validation ...")
    try:
        TweetSentimentRawSchema.validate(df, lazy=True)
        print("          PASS")
    except pa.errors.SchemaErrors as exc:
        for _, row in exc.failure_cases.iterrows():
            failures.append(
                f"[L1/L2] {row['column']}: {row['check']} "
                f"(failed on {row['failure_case']!r})"
            )
        print(f"          FAIL -- {len(exc.failure_cases)} case(s)")

    # ---------------- LEVEL 3: statistical --------------------------------
    print("  [L3]    statistical validation ...")
    l3 = check_uniqueness(df, data_cfg["id_column"])
    if schema_cfg.get("expected_label_proportions"):
        l3 += check_label_distribution(
            df,
            data_cfg["label_column"],
            schema_cfg["expected_label_proportions"],
            schema_cfg["label_proportion_tolerance"],
        )
    failures += l3
    print("          PASS" if not l3 else f"          FAIL -- {len(l3)} issue(s)")

    # ---------------- LEVEL 4: business rules -----------------------------
    print("  [L4]    business rule validation ...")
    findings += check_fabricated_cycle(
        df,
        schema_cfg.get("fabricated_cycle_column"),
        schema_cfg.get("fabricated_cycle_length"),
    )
    findings += check_label_span_consistency(df)
    findings += check_leading_whitespace(df, data_cfg["text_column"])

    null_text = int(df[data_cfg["text_column"]].isnull().sum())
    if null_text:
        findings.append(
            f"[L4/BR-04] {null_text} row(s) have a null "
            f"'{data_cfg['text_column']}'. These are dropped during cleaning "
            f"and the drop is logged -- never silently filtered."
        )
    print(f"          {len(findings)} finding(s)")

    # ---------------- report ----------------------------------------------
    print("\n" + "-" * 72)
    if failures:
        print(f"  BLOCKING FAILURES ({len(failures)}):")
        for f in failures:
            print(f"    x {f}")
    if findings:
        print(f"\n  FINDINGS ({len(findings)}) -- handled by configuration:")
        for f in findings:
            print(f"    ! {f}")
    print("-" * 72)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "task": cfg["task"]["name"],
                "source": src_path,
                "rows": len(df),
                "columns": len(df.columns),
                "status": "FAIL" if failures else "PASS",
                "blocking_failures": failures,
                "findings": findings,
            },
            fh,
            indent=2,
        )
    print(f"  report written: {REPORT_PATH}")

    if failures:
        print("\n  RESULT: FAIL -- pipeline halted. Downstream stages will not run.")
        return 1

    print(f"\n  RESULT: PASS -- {len(df):,} rows validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
