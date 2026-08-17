"""
Tests for the drift detector.

The critical test is the FIRST one. A detector that fires on clean traffic is
worse than no detector: it trains the operator to ignore alerts, and then the
real alert is ignored too. Everything else is secondary to that.
"""

from __future__ import annotations

import sqlite3

import joblib
import pytest
import yaml

from src.features.text_features import load_bundle
from src.monitoring.drift import evaluate_window
from src.monitoring.simulate import build_small_window, build_windows


@pytest.fixture(scope="module")
def setup():
    cfg = yaml.safe_load(open("configs/sentiment.yaml", encoding="utf-8"))
    bundle = load_bundle("models/feature_bundle.joblib")
    model = joblib.load("models/classifier.joblib")
    with sqlite3.connect("data/processed/feature_store.db") as conn:
        baseline = [r[0] for r in conn.execute("SELECT text FROM features_test")]
    return cfg, bundle, model, baseline


def test_control_window_raises_no_alert(setup):
    """
    THE FALSE-POSITIVE TEST.

    The control window is a resample of the baseline pool, so it is undrifted by
    construction. Any breach here is a threshold defect.
    """
    cfg, bundle, model, baseline = setup
    control = build_windows(baseline)["W0_control_no_drift"]
    result = evaluate_window("control", baseline, control, model, bundle,
                             cfg["monitoring"])
    assert result["n_breached"] == 0, f"false positive: {result['signals_breached']}"
    assert result["decision"] == "NO_ACTION"


def test_vocabulary_drift_is_detected(setup):
    cfg, bundle, model, baseline = setup
    w = build_windows(baseline)["W1_vocabulary_drift"]
    result = evaluate_window("vocab", baseline, w, model, bundle, cfg["monitoring"])
    assert "oov_rate" in result["signals_breached"]
    assert result["decision"] == "RETRAIN"


def test_topic_drift_is_detected(setup):
    cfg, bundle, model, baseline = setup
    w = build_windows(baseline)["W2_topic_drift"]
    result = evaluate_window("topic", baseline, w, model, bundle, cfg["monitoring"])
    assert result["n_breached"] >= 2
    assert result["decision"] == "RETRAIN"


def test_format_drift_fires_length_but_not_oov(setup):
    """
    Signals must discriminate between KINDS of drift, not merely detect that
    something changed. Concatenating existing texts changes length without
    introducing new vocabulary, so OOV must stay quiet.
    """
    cfg, bundle, model, baseline = setup
    w = build_windows(baseline)["W3_format_drift"]
    result = evaluate_window("format", baseline, w, model, bundle, cfg["monitoring"])
    assert "doc_length_sigma" in result["signals_breached"]
    assert "oov_rate" not in result["signals_breached"]


def test_insufficient_samples_holds_rather_than_retrains(setup):
    """
    Detection and action are different problems. Retraining on 120 examples
    would fit the drift's noise rather than correct for it.
    """
    cfg, bundle, model, baseline = setup
    small = build_small_window(baseline)
    result = evaluate_window("small", baseline, small, model, bundle, cfg["monitoring"])
    assert result["n_breached"] >= 2
    assert result["decision"] == "HOLD"


def test_confidence_signal_respects_the_effect_floor(setup):
    """
    A statistically significant but operationally trivial shift must not fire.
    Measured on the format window: p=0.0025 with a mean delta of only -0.009.
    """
    cfg, bundle, model, baseline = setup
    w = build_windows(baseline)["W3_format_drift"]
    result = evaluate_window("format", baseline, w, model, bundle, cfg["monitoring"])
    conf = next(s for s in result["signals"] if s["signal"] == "confidence_ks")
    assert conf["statistically_significant"] is True
    assert conf["materially_large"] is False
    assert conf["breached"] is False
