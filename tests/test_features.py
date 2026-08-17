"""
Unit tests for the shared feature module.

The instructor's own 60-minute tutorial ends with a governance checklist whose
last three items are UNTICKED:

    [ x ] Automated retraining pipeline   -- manual for now
    [ x ] Unit tests on feature pipeline  -- deferred
    [ x ] Docker containerisation         -- deferred

This file closes the second one. The tests below are not decorative: each
asserts a property whose violation would cause a silent, HTTP-200 production
failure that no amount of monitoring on accuracy alone would catch quickly.

Run:
    pytest tests/test_features.py -v
"""

from __future__ import annotations

import pytest

from src.features.text_features import (
    fit_vectorizer,
    load_bundle,
    normalise_text,
    save_bundle,
)

FEATURES_CFG = {
    "lowercase": True,
    "ngram_range": [1, 2],
    "max_features": 500,
    "min_df": 1,
    "sublinear_tf": True,
    "stop_words": None,
}

CORPUS = [
    " this film was a complete waste of time",
    "an absolute masterpiece, highly recommend",
    "it was fine, nothing special",
    "terrible acting and a dull script",
    "beautiful cinematography and a strong cast",
]


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def test_normalise_strips_leading_whitespace():
    """BR-03: 39.9% of training rows had leading whitespace."""
    assert normalise_text("  hello world") == "hello world"


def test_normalise_strips_trailing_whitespace():
    assert normalise_text("hello world   ") == "hello world"


def test_normalise_coerces_non_string():
    """Serving receives JSON; a client may send a number or a bool."""
    assert normalise_text(42) == "42"
    assert normalise_text(True) == "True"


def test_normalise_is_idempotent():
    """
    Applying normalisation twice must equal applying it once.

    Ingestion strips whitespace, and the feature module strips again as defence
    in depth. If normalisation were not idempotent, that second application
    would alter already-clean training data and reintroduce the very skew the
    two layers exist to prevent.
    """
    once = normalise_text("  spaced out  ")
    assert normalise_text(once) == once


# ---------------------------------------------------------------------------
# The training-serving contract
# ---------------------------------------------------------------------------

def test_transform_is_deterministic():
    """The same input must produce the same vector. Every time."""
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    a = bundle.transform(["a dull script"])
    b = bundle.transform(["a dull script"])
    assert (a != b).nnz == 0


def test_leading_whitespace_does_not_change_the_vector():
    """
    THE SKEW TEST.

    If training strips whitespace and serving does not, these two inputs
    produce different vectors and every affected prediction shifts. Because
    both paths call the same normalise_text, they cannot.
    """
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    clean = bundle.transform(["a dull script"])
    dirty = bundle.transform(["   a dull script   "])
    assert (clean != dirty).nnz == 0


def test_feature_count_is_stable_across_calls():
    """
    Serving must always emit vectors of the width the model expects.

    An unseen word must NOT widen the matrix -- it must be dropped as OOV.
    A width change would raise a shape error at predict time in the best case,
    and silently misalign coefficients in the worst.
    """
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    n = bundle.n_features
    assert bundle.transform(["a dull script"]).shape[1] == n
    assert bundle.transform(["zzzz unseen vocabulary xyzzy"]).shape[1] == n


def test_roundtrip_preserves_the_vector(tmp_path):
    """
    Persisting and reloading the bundle must not change its behaviour.

    This is the test that guards the actual deployment path: the vectorizer is
    fitted in the training process and reloaded in a different process, on a
    different machine, possibly weeks later.
    """
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    before = bundle.transform(["a dull script"])

    path = tmp_path / "bundle.joblib"
    save_bundle(bundle, path)
    reloaded = load_bundle(path)

    after = reloaded.transform(["a dull script"])
    assert (before != after).nnz == 0
    assert reloaded.n_features == bundle.n_features
    assert reloaded.vocabulary_size == bundle.vocabulary_size


def test_bundle_exposes_no_fit_method():
    """
    Serving must not be able to re-fit the vectorizer, even by accident.

    FeatureBundle deliberately exposes transform() and not fit(). This asserts
    that design property rather than trusting it.
    """
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    assert not hasattr(bundle, "fit")
    assert not hasattr(bundle, "fit_transform")


# ---------------------------------------------------------------------------
# Edge cases the serving layer will actually receive
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "",                       # empty string
        "   ",                    # whitespace only
        "!!!???",                 # punctuation only
        "\U0001F600\U0001F600",   # emoji only
        "a",                      # single character
        "word " * 2000,           # very long input
    ],
)
def test_transform_survives_degenerate_input(text):
    """
    None of these may raise. An all-zero vector is an acceptable answer;
    a 500 from the API is not.
    """
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    vec = bundle.transform([text])
    assert vec.shape == (1, bundle.n_features)


def test_oov_rate_on_training_corpus_is_zero():
    """Every token in the fitted corpus is in the vocabulary by construction."""
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    assert bundle.oov_rate(CORPUS) == pytest.approx(0.0)


def test_oov_rate_on_unseen_vocabulary_is_one():
    """The M5 drift signal must respond to genuinely novel language."""
    bundle = fit_vectorizer(CORPUS, FEATURES_CFG)
    assert bundle.oov_rate(["qwertyuiop asdfghjkl zxcvbnm"]) == pytest.approx(1.0)
