"""
Drift detection.

Usage:
    python -m src.monitoring.drift --config configs/sentiment.yaml

--------------------------------------------------------------------------
WHY DRIFT DETECTION ON TEXT IS HARDER THAN ON TABULAR DATA
--------------------------------------------------------------------------
The course tutorial monitors a tabular feature by comparing its mean against a
training baseline: if distance_km shifts by more than 1 sigma, investigate.

That has no direct analogue here. You cannot take the mean of a sentence. So
four proxy signals are used instead, each mapped to a technique the course
prescribes in M2 2.5.3:

  1. OOV RATE                proportion test    new words entering the traffic
  2. TOKEN FREQUENCY         chi-squared        the vocabulary MIX shifting
  3. PREDICTION CONFIDENCE   KS test            the model becoming less certain
  4. DOCUMENT LENGTH         sigma shift        input shape changing

No single signal is sufficient:

  - OOV rate rises for a harmless reason (a new product name) as well as a
    harmful one (a new topic the model has never seen).
  - Confidence can fall while accuracy holds, and can hold while accuracy
    falls.
  - Document length can shift with no semantic change at all -- a new client
    integration that pads its inputs.

Hence the retraining trigger requires TWO OR MORE signals breached. One signal
is noise; two is evidence. That threshold is a design decision and is justified
in docs/MONITORING.md rather than asserted here.

--------------------------------------------------------------------------
WHAT THIS MODULE DOES NOT DO
--------------------------------------------------------------------------
It does not measure accuracy. In production the true labels are not available
at inference time -- that is the entire reason drift detection exists. If
labels were available you would simply compute accuracy and skip all of this.

Every signal below is therefore computed WITHOUT ground truth. That constraint
is the discipline: a monitor that needs labels is not a monitor, it is an
evaluation.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import yaml
from scipy import stats

from src.features.text_features import load_bundle, normalise_text

BUNDLE_PATH = Path("models/feature_bundle.joblib")
MODEL_PATH = Path("models/classifier.joblib")
STORE_PATH = Path("data/processed/feature_store.db")
MANIFEST_PATH = Path("reports/feature_manifest.json")
OUT_PATH = Path("reports/drift_report.json")


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Signal 1 -- OOV rate
# ---------------------------------------------------------------------------

def signal_oov(bundle, texts: list[str], threshold: float) -> dict:
    """
    Share of unigram tokens absent from the fitted vocabulary.

    Restricted to unigrams. Measured earlier on this corpus: bigram-inclusive
    OOV sits at 0.32 on held-out data purely because most word PAIRS are unseen
    even when every word is familiar. A signal already near saturation cannot
    indicate a change, so bigrams are excluded from the measurement (though
    they remain in the model's feature space).
    """
    rate = bundle.oov_rate(texts)
    return {
        "signal": "oov_rate",
        "value": round(rate, 4),
        "threshold": threshold,
        "breached": bool(rate > threshold),
        "interpretation": (
            "new vocabulary entering the traffic"
            if rate > threshold
            else "vocabulary consistent with training"
        ),
    }


# ---------------------------------------------------------------------------
# Signal 2 -- token frequency chi-squared
# ---------------------------------------------------------------------------

def signal_token_chi2(
    bundle, baseline: list[str], current: list[str], top_k: int, p_threshold: float
) -> dict:
    """
    Chi-squared test on the top-K token frequency distribution.

    M2 2.5.3 prescribes chi-squared for categorical distributions, and a
    vocabulary is categorical. This catches a shift in the token MIX even when
    every token is in-vocabulary -- which is precisely the case OOV rate misses.

    Counts are added to both arms (+1) to avoid zero expected frequencies, which
    would make the statistic undefined.
    """
    analyzer = bundle.vectorizer.build_analyzer()

    def unigrams(texts):
        c = Counter()
        for t in texts:
            c.update(tok for tok in analyzer(normalise_text(t)) if " " not in tok)
        return c

    base_counts, cur_counts = unigrams(baseline), unigrams(current)
    vocab = [w for w, _ in base_counts.most_common(top_k)]
    if not vocab:
        return {
            "signal": "token_frequency_chi2",
            "value": None,
            "breached": False,
            "interpretation": "insufficient tokens to test",
        }

    observed = np.array([cur_counts.get(w, 0) for w in vocab], dtype=float) + 1
    expected = np.array([base_counts[w] for w in vocab], dtype=float) + 1
    # Scale expected to the observed total so the test measures SHAPE, not volume.
    expected = expected * (observed.sum() / expected.sum())

    stat, p = stats.chisquare(observed, expected)
    return {
        "signal": "token_frequency_chi2",
        "value": round(float(p), 6),
        "statistic": round(float(stat), 2),
        "top_k": top_k,
        "threshold": p_threshold,
        "breached": bool(p < p_threshold),
        "interpretation": (
            "token mix has shifted significantly"
            if p < p_threshold
            else "token mix consistent with baseline"
        ),
    }


# ---------------------------------------------------------------------------
# Signal 3 -- prediction confidence KS test
# ---------------------------------------------------------------------------

def signal_confidence_ks(
    model, bundle, baseline: list[str], current: list[str],
    p_threshold: float, min_effect: float,
) -> dict:
    """
    Two-sample Kolmogorov-Smirnov test on the predict_proba max distribution,
    gated by a minimum effect size.

    M2 2.5.3 prescribes KS for continuous distributions. Confidence is
    continuous, requires no labels, and falls when inputs move away from the
    training manifold -- so it is the closest label-free proxy for accuracy
    degradation available.

    ------------------------------------------------------------------
    WHY A P-VALUE ALONE IS NOT ENOUGH -- an empirical finding
    ------------------------------------------------------------------
    The first version of this signal fired on p < 0.01 alone. On the format-drift
    window that produced:

        p = 0.0025  (significant)   mean confidence 0.631 -> 0.622

    A 0.009 shift in mean confidence is operationally meaningless, but with
    n=600 against a 5,496-row baseline the KS test has enough power to call it
    significant. That is the well-known gap between STATISTICAL and PRACTICAL
    significance, and left unguarded it would generate steady false alarms in
    production -- where sample sizes are larger still and the test more
    sensitive.

    The same lesson appeared in Sprint 2: a 0.002 difference in f1_macro between
    two runs was statistically detectable and operationally irrelevant. There the
    remedy was a bootstrap confidence interval; here it is an effect-size floor.

    The signal now requires BOTH:
        p < p_threshold                       the shift is real, and
        |mean change| >= min_effect           the shift is large enough to care

    A significant-but-tiny shift is reported as observed, with breached=False.
    """
    base_conf = model.predict_proba(bundle.transform(baseline)).max(axis=1)
    cur_conf = model.predict_proba(bundle.transform(current)).max(axis=1)
    stat, p = stats.ks_2samp(base_conf, cur_conf)

    delta = float(cur_conf.mean() - base_conf.mean())
    significant = bool(p < p_threshold)
    material = bool(abs(delta) >= min_effect)

    if significant and material:
        interpretation = (
            f"confidence shifted materially "
            f"({base_conf.mean():.3f} -> {cur_conf.mean():.3f}, "
            f"delta {delta:+.3f})"
        )
    elif significant and not material:
        interpretation = (
            f"statistically significant but below the {min_effect} effect floor "
            f"(delta {delta:+.3f}) -- not actioned"
        )
    else:
        interpretation = "confidence distribution stable"

    return {
        "signal": "confidence_ks",
        "value": round(float(p), 6),
        "statistic": round(float(stat), 4),
        "baseline_mean_confidence": round(float(base_conf.mean()), 4),
        "current_mean_confidence": round(float(cur_conf.mean()), 4),
        "delta": round(delta, 4),
        "threshold": p_threshold,
        "min_effect": min_effect,
        "statistically_significant": significant,
        "materially_large": material,
        "breached": bool(significant and material),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Signal 4 -- document length sigma shift
# ---------------------------------------------------------------------------

def signal_doc_length(
    baseline: list[str], current: list[str], investigate: float, halt: float
) -> dict:
    """
    Sigma shift in mean token count -- the one signal with a direct tabular
    analogue, and the one that mirrors the tutorial's threshold scheme:
    beyond 1 sigma investigate, beyond 2 sigma stop the pipeline.
    """
    base_len = np.array([len(normalise_text(t).split()) for t in baseline], dtype=float)
    cur_len = np.array([len(normalise_text(t).split()) for t in current], dtype=float)

    mu, sigma = base_len.mean(), base_len.std()
    shift = abs(cur_len.mean() - mu) / sigma if sigma > 0 else 0.0

    if shift > halt:
        level = "HALT"
    elif shift > investigate:
        level = "INVESTIGATE"
    else:
        level = "OK"

    return {
        "signal": "doc_length_sigma",
        "value": round(float(shift), 4),
        "baseline_mean_tokens": round(float(mu), 2),
        "current_mean_tokens": round(float(cur_len.mean()), 2),
        "baseline_sigma": round(float(sigma), 2),
        "threshold_investigate": investigate,
        "threshold_halt": halt,
        "level": level,
        "breached": bool(shift > investigate),
        "interpretation": (
            f"mean length moved {shift:.2f} sigma ({mu:.1f} -> {cur_len.mean():.1f} tokens)"
        ),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def evaluate_window(
    label: str, baseline: list[str], current: list[str], model, bundle, mon_cfg: dict
) -> dict:
    cfg = mon_cfg["drift_signals"]
    signals = [
        signal_oov(bundle, current, cfg["oov_rate"]["threshold"]),
        signal_token_chi2(
            bundle, baseline, current,
            cfg["token_frequency_chi2"]["top_k"],
            cfg["token_frequency_chi2"]["p_value_threshold"],
        ),
        signal_confidence_ks(
            model, bundle, baseline, current,
            cfg["confidence_ks"]["p_value_threshold"],
            cfg["confidence_ks"]["min_effect_size"],
        ),
        signal_doc_length(
            baseline, current,
            cfg["doc_length_sigma"]["investigate"],
            cfg["doc_length_sigma"]["halt"],
        ),
    ]

    breached = [s for s in signals if s["breached"]]
    trig = mon_cfg["retraining_trigger"]
    n_required = trig["min_signals_breached"]
    min_samples = trig["min_new_samples"]

    enough_samples = len(current) >= min_samples
    fire = len(breached) >= n_required and enough_samples

    if fire:
        decision = "RETRAIN"
        rationale = (
            f"{len(breached)} of 4 signals breached (>= {n_required} required) "
            f"and {len(current)} samples available (>= {min_samples} required)"
        )
    elif len(breached) >= n_required and not enough_samples:
        decision = "HOLD"
        rationale = (
            f"{len(breached)} signals breached but only {len(current)} samples "
            f"(< {min_samples}). Retraining on too little data would overfit "
            f"the drift instead of correcting for it."
        )
    elif len(breached) == 1:
        decision = "MONITOR"
        rationale = (
            f"1 signal breached ({breached[0]['signal']}). A single signal is "
            f"treated as noise; continue observing."
        )
    else:
        decision = "NO_ACTION"
        rationale = "No signals breached. Traffic consistent with training."

    return {
        "window": label,
        "n_baseline": len(baseline),
        "n_current": len(current),
        "signals": signals,
        "signals_breached": [s["signal"] for s in breached],
        "n_breached": len(breached),
        "decision": decision,
        "rationale": rationale,
    }


def print_window(w: dict) -> None:
    print(f"\n  {'-'*68}")
    print(f"  WINDOW: {w['window']}   (n={w['n_current']:,} vs baseline n={w['n_baseline']:,})")
    print(f"  {'-'*68}")
    for s in w["signals"]:
        mark = "BREACH" if s["breached"] else "  ok  "
        val = s["value"] if s["value"] is not None else "n/a"
        print(f"    [{mark}] {s['signal']:22s} {str(val):>10s}   {s['interpretation']}")
    print(f"\n    signals breached : {w['n_breached']} of 4")
    print(f"    DECISION         : {w['decision']}")
    print(f"    rationale        : {w['rationale']}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Drift detection")
    ap.add_argument("--config", required=True)
    ap.add_argument(
        "--from-log",
        action="store_true",
        help="Score the live prediction log instead of the simulation windows",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    mon_cfg = cfg["monitoring"]

    if not BUNDLE_PATH.exists() or not MODEL_PATH.exists():
        print("  FAIL: artefacts missing. Run dvc repro + export_model first.")
        return 1

    bundle = load_bundle(BUNDLE_PATH)
    model = joblib.load(MODEL_PATH)

    print("=" * 72)
    print("  DRIFT DETECTION")
    print("=" * 72)

    # Baseline: the held-out test split. Same distribution as training, never
    # trained on -- so it represents "traffic that has not drifted".
    with sqlite3.connect(STORE_PATH) as conn:
        rows = conn.execute("SELECT text FROM features_test").fetchall()
    baseline = [r[0] for r in rows]
    print(f"  baseline reference: {len(baseline):,} held-out texts")

    windows = []

    if args.from_log:
        log_db = Path("data/processed/predictions.db")
        if not log_db.exists():
            print("  FAIL: no prediction log. Send traffic to the API first.")
            return 1
        with sqlite3.connect(log_db) as conn:
            live = [r[0] for r in conn.execute("SELECT text FROM predictions").fetchall()]
        print(f"  live traffic: {len(live):,} logged predictions")
        windows.append(evaluate_window("live_traffic", baseline, live, model, bundle, mon_cfg))
    else:
        from src.monitoring.simulate import build_small_window, build_windows

        for name, texts in build_windows(baseline).items():
            windows.append(evaluate_window(name, baseline, texts, model, bundle, mon_cfg))

        # W4 exercises the HOLD path: real drift, insufficient data to act on.
        windows.append(
            evaluate_window(
                "W4_drift_below_sample_minimum",
                baseline,
                build_small_window(baseline),
                model, bundle, mon_cfg,
            )
        )

    for w in windows:
        print_window(w)

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "model_version": cfg["serving"]["model_version"],
                "trigger_policy": mon_cfg["retraining_trigger"],
                "windows": windows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  written: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
