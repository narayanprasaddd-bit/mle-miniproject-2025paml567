"""
Four-level data validation for the text classification pipeline.

Implements the validation framework from M2 Chapter 2.5:

    Level 1  Schema      -- column names, dtypes, required fields
    Level 2  Range/Domain -- semantic validity of values
    Level 3  Statistical  -- distribution vs. training baseline
    Level 4  Business rule -- compound, domain-specific constraints

Design principle (M2 2.5.4): validation MUST fail loudly. Every check either
passes or raises. Nothing warns-and-continues, because a pipeline that
continues after a validation failure propagates corrupt data to every
downstream component and makes the failure far harder to diagnose.

Pandera is configured with coerce=False throughout. Silent type coercion is
exactly the failure mode that caused the StreamSphere incident in Lab 2:
total_charges arrived as the string "480.50", was coerced to NaN, and the model
served wrong predictions for three days without a single error.
"""

from __future__ import annotations

import pandas as pd

# pandera >=0.23 moved the pandas API to a submodule and warns on the old path.
# This shim keeps the module working on both pinned (0.20) and newer versions.
try:
    import pandera.pandas as pa
    from pandera.pandas import Series
except (ImportError, AttributeError):  # pandera < 0.23
    import pandera as pa
    from pandera.typing import Series


class ValidationError(Exception):
    """Raised when any validation level fails. Callers exit(1) on this."""


# ---------------------------------------------------------------------------
# LEVEL 1 + 2 -- Schema and range/domain validation, declaratively
# ---------------------------------------------------------------------------

class TweetSentimentRawSchema(pa.DataFrameModel):
    """
    Schema contract for the raw Tweet Sentiment Extraction training file.

    Version-controlled alongside the feature engineering code, as M2 2.5.1
    requires: any change to this schema must go through code review, because a
    schema change is a change to the model's input contract.
    """

    textID: Series[str] = pa.Field(unique=True, nullable=False)
    text: Series[str] = pa.Field(nullable=True)  # 1 known null -- handled in cleaning
    selected_text: Series[str] = pa.Field(nullable=True)
    sentiment: Series[str] = pa.Field(isin=["negative", "neutral", "positive"])

    # Fabricated metadata columns. Declared so that schema validation still
    # passes on the file as distributed, but excluded from features by config.
    # Declaring them is not endorsing them -- see business rule BR-01.
    Time_of_Tweet: Series[str] = pa.Field(
        alias="Time of Tweet", isin=["morning", "noon", "night"]
    )
    Age_of_User: Series[str] = pa.Field(
        alias="Age of User",
        isin=["0-20", "21-30", "31-45", "46-60", "60-70", "70-100"],
    )

    class Config:
        coerce = False        # fail explicitly rather than silently convert
        strict = False        # tolerate the extra fabricated geo columns
        add_missing_columns = False


class TweetSentimentCleanSchema(pa.DataFrameModel):
    """Schema contract AFTER cleaning. Nulls are no longer tolerated."""

    textID: Series[str] = pa.Field(unique=True, nullable=False)
    text: Series[str] = pa.Field(nullable=False, str_length={"min_value": 1})
    sentiment: Series[str] = pa.Field(isin=["negative", "neutral", "positive"])

    class Config:
        coerce = False
        strict = False


# ---------------------------------------------------------------------------
# LEVEL 3 -- Statistical validation against the training baseline
# ---------------------------------------------------------------------------

def check_label_distribution(
    df: pd.DataFrame,
    label_column: str,
    expected: dict[str, float],
    tolerance: float,
) -> list[str]:
    """
    Compare observed label proportions against the recorded training baseline.

    A shift here does not necessarily mean corrupt data -- it may mean genuine
    population change, which is a drift signal rather than a defect. Either way
    it must surface at the ingestion boundary rather than during model debugging.
    """
    errors: list[str] = []
    observed = df[label_column].value_counts(normalize=True).to_dict()

    for label, expected_prop in expected.items():
        actual_prop = observed.get(label, 0.0)
        delta = abs(actual_prop - expected_prop)
        if delta > tolerance:
            errors.append(
                f"[L3] label '{label}' proportion {actual_prop:.4f} deviates "
                f"{delta:.4f} from baseline {expected_prop:.4f} "
                f"(tolerance {tolerance})"
            )
    return errors


def check_uniqueness(df: pd.DataFrame, id_column: str) -> list[str]:
    """
    M2 2.2.6: duplicates inflate patterns relative to their true frequency and
    bias model training. Detected by comparing distinct key count to row count.

    This is the check that caught the corrupted Sentiment140 export.
    """
    errors: list[str] = []
    n_rows, n_unique = len(df), df[id_column].nunique()
    if n_rows != n_unique:
        errors.append(
            f"[L3] uniqueness violation on '{id_column}': "
            f"{n_rows} rows but {n_unique} distinct keys "
            f"({n_rows - n_unique} duplicates)"
        )
    return errors


# ---------------------------------------------------------------------------
# LEVEL 4 -- Business rule validation
# ---------------------------------------------------------------------------

def check_fabricated_cycle(
    df: pd.DataFrame,
    column: str,
    cycle_length: int,
) -> list[str]:
    """
    BR-01 -- Fabricated column detection.

    A genuine per-record attribute does not repeat on a fixed row cycle. If
    column[i] == column[i + N] for all i, the column was appended by position
    rather than joined by key, and therefore carries no information about the
    record it sits beside.

    This rule was written in response to a real finding in this dataset: the
    'Country' column cycles alphabetically through 195 countries and repeats
    every 195 rows (row 0 == row 195 == 'Afghanistan'). See
    docs/DATA_QUALITY_INCIDENT.md section 3.

    Returns a finding, not a fatal error -- the correct response is to exclude
    the column from features, which configs/*.yaml already does.
    """
    findings: list[str] = []
    if column not in df.columns or cycle_length is None:
        return findings
    if len(df) <= cycle_length:
        return findings

    values = df[column].tolist()
    n_compare = min(len(values) - cycle_length, cycle_length * 3)
    matches = sum(
        1 for i in range(n_compare) if values[i] == values[i + cycle_length]
    )

    if n_compare > 0 and matches == n_compare:
        findings.append(
            f"[L4/BR-01] column '{column}' repeats exactly on a "
            f"{cycle_length}-row cycle across {n_compare} compared positions. "
            f"Column is positionally appended, not key-joined: it carries no "
            f"per-record information and MUST be excluded from features."
        )
    return findings


def check_label_span_consistency(df: pd.DataFrame) -> list[str]:
    """
    BR-02 -- selected_text must be a substring of text.

    M2 2.5.4 notes that business rule violations often indicate LABEL quality
    problems. 'selected_text' is the annotated span carrying the sentiment; if
    it is not contained in 'text', the annotation does not correspond to the
    record and the label is suspect.

    Reported as a rate rather than a hard failure: some mismatch is expected
    from whitespace and encoding artefacts in the source annotation process.
    """
    findings: list[str] = []
    if "selected_text" not in df.columns:
        return findings

    sub = df.dropna(subset=["text", "selected_text"])
    if len(sub) == 0:
        return findings

    contained = [
        str(s).strip() in str(t) for t, s in zip(sub["text"], sub["selected_text"])
    ]
    mismatch_rate = 1.0 - (sum(contained) / len(contained))

    if mismatch_rate > 0.05:
        findings.append(
            f"[L4/BR-02] {mismatch_rate:.2%} of rows have a 'selected_text' "
            f"that is not a substring of 'text'. Annotation quality is suspect "
            f"for those rows."
        )
    return findings


def check_leading_whitespace(df: pd.DataFrame, text_column: str) -> list[str]:
    """
    BR-03 -- Training/serving skew vector.

    ~40% of rows in this dataset begin with a leading space. That is harmless
    IF and ONLY IF the same normalisation runs at training and at serving time.
    If training strips and serving does not (or vice versa), the tokenisation
    differs and every prediction shifts silently.

    Surfaced as a finding so that the shared feature module's strip step is
    justified by evidence rather than habit.
    """
    findings: list[str] = []
    sub = df.dropna(subset=[text_column])
    if len(sub) == 0:
        return findings

    rate = sub[text_column].astype(str).str.startswith(" ").mean()
    if rate > 0.01:
        findings.append(
            f"[L4/BR-03] {rate:.1%} of '{text_column}' values have leading "
            f"whitespace. The shared feature module MUST strip identically at "
            f"training and serving time (skew vector)."
        )
    return findings
