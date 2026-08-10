"""
Negative test for the validation pipeline.

Writes a deliberately corrupted copy of the raw training data so that the
validation gate can be proven to FAIL. A validation suite that has only ever
been run against clean data is untested: it demonstrates that nothing was
detected, not that detection works.

Usage:
    python -m tests.inject_defects
    python -m src.validation.validate --config configs/sentiment.yaml \
        --input data/raw/_corrupt_test.csv     # expect exit code 1

Injected defects, one per validation level:
    L1/L2  invalid category in 'sentiment'      -> domain violation
    L1/L2  invalid category in 'Time of Tweet'  -> enum violation
    L3     duplicated textID                    -> uniqueness violation
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SRC = Path("data/raw/train.csv")
DST = Path("data/raw/_corrupt_test.csv")


def main() -> int:
    if not SRC.exists():
        print(f"  source not found: {SRC}")
        return 1

    df = pd.read_csv(SRC, encoding="latin-1")

    df.loc[5, "sentiment"] = "POSITIVE!!"        # L1/L2 domain violation
    df.loc[9, "Time of Tweet"] = "midnight"      # L1/L2 enum violation
    df.loc[0, "textID"] = df.loc[1, "textID"]    # L3 uniqueness violation

    df.to_csv(DST, index=False, encoding="latin-1")

    print(f"  wrote {DST} with 3 injected defects:")
    print("    row 5  sentiment      -> 'POSITIVE!!'  (L1/L2)")
    print("    row 9  Time of Tweet  -> 'midnight'    (L1/L2)")
    print("    row 0  textID         -> duplicate     (L3)")
    print("\n  now run:")
    print("    python -m src.validation.validate --config configs/sentiment.yaml \\")
    print(f"        --input {DST}")
    print("  expected: exit code 1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
