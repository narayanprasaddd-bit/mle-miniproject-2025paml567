"""
API contract tests.

Run:
    pytest tests/test_api.py -v

--------------------------------------------------------------------------
WHAT IS BEING TESTED
--------------------------------------------------------------------------
The brief's M4 task is explicit: "handle malformed/edge-case inputs." These
tests assert that requirement rather than assuming it.

The governing distinction throughout:

    422  the CLIENT sent something invalid   -> the service is working
    500  the SERVICE broke                   -> the service is not working

Conflating them makes production debugging significantly harder: a 500 sends an
engineer looking for a bug that does not exist. So every malformed-input test
below asserts 422 specifically, not merely "an error".
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.serving.app import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def test_health_returns_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_confirms_both_artefacts_loaded(client):
    """
    A service with a model but no vectorizer starts cleanly and then produces
    silent nonsense. Both must be reported.
    """
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["vectorizer_loaded"] is True
    assert body["n_features"] > 0
    assert body["vocabulary_size"] > 0


def test_health_reports_the_three_classes(client):
    assert sorted(client.get("/health").json()["classes"]) == [
        "negative",
        "neutral",
        "positive",
    ]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_predict_returns_a_valid_class(client):
    r = client.post("/predict/sentiment", json={"text": "this was a total waste of time"})
    assert r.status_code == 200
    assert r.json()["predicted"] in {"negative", "neutral", "positive"}


def test_probabilities_sum_to_one(client):
    body = client.post(
        "/predict/sentiment", json={"text": "an absolute masterpiece"}
    ).json()
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)


def test_confidence_matches_the_winning_probability(client):
    """Internal consistency: confidence must BE the max probability."""
    body = client.post("/predict/sentiment", json={"text": "it was fine"}).json()
    assert body["confidence"] == pytest.approx(
        max(body["probabilities"].values()), abs=1e-3
    )


def test_response_carries_the_model_version(client):
    """Untraceable predictions cannot be audited after an incident."""
    assert client.post(
        "/predict/sentiment", json={"text": "good"}
    ).json()["model_version"]


def test_low_confidence_flag_is_consistent_with_the_threshold(client):
    body = client.post("/predict/sentiment", json={"text": "the thing happened"}).json()
    assert body["low_confidence"] == (body["confidence"] < 0.50)


# ---------------------------------------------------------------------------
# Malformed input -- must be 422, never 500
# ---------------------------------------------------------------------------

def test_missing_field_returns_422(client):
    assert client.post("/predict/sentiment", json={}).status_code == 422


def test_wrong_field_name_returns_422(client):
    assert client.post("/predict/sentiment", json={"txt": "hello"}).status_code == 422


def test_wrong_type_returns_422(client):
    assert client.post("/predict/sentiment", json={"text": 12345}).status_code == 422


def test_null_returns_422(client):
    assert client.post("/predict/sentiment", json={"text": None}).status_code == 422


def test_list_instead_of_string_returns_422(client):
    assert client.post("/predict/sentiment", json={"text": ["a", "b"]}).status_code == 422


def test_empty_string_returns_422(client):
    assert client.post("/predict/sentiment", json={"text": ""}).status_code == 422


def test_whitespace_only_returns_422(client):
    """
    min_length=1 accepts "   " -- the custom validator rejects it.

    Without this, an all-zero feature vector would still produce a
    confident-looking label for input containing no information at all.
    """
    r = client.post("/predict/sentiment", json={"text": "     "})
    assert r.status_code == 422


def test_oversized_text_returns_422_not_500(client):
    """A 10,000-character input must be refused politely, not crash the service."""
    r = client.post("/predict/sentiment", json={"text": "word " * 2000})
    assert r.status_code == 422


def test_malformed_json_returns_422(client):
    r = client.post(
        "/predict/sentiment",
        content=b"{not valid json",
        headers={"Content-Type": "application/json"},
    )
    assert r.status_code == 422


# ---------------------------------------------------------------------------
# Edge cases that must SUCCEED -- unusual is not invalid
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    [
        "a",                                  # single character
        "!!!???...",                          # punctuation only
        "\U0001F600 \U0001F602",              # emoji only
        "asdfghjkl qwertyuiop zxcvbnm",        # entirely out-of-vocabulary
        "CAPS LOCK SHOUTING",                 # casing
        "caf\u00e9 na\u00efve \u00fcber",     # accented, non-ASCII
        "  leading and trailing  ",            # whitespace the module strips
        "\u092f\u0939 \u0905\u0915\u094d\u0937\u0930",  # non-Latin script
    ],
)
def test_degenerate_but_valid_input_returns_200(client, text):
    """
    None of these may 500. An all-zero vector yielding the majority class is an
    acceptable answer; a stack trace is not.
    """
    r = client.post("/predict/sentiment", json={"text": text})
    assert r.status_code == 200, f"failed on {text!r}"
    assert r.json()["predicted"] in {"negative", "neutral", "positive"}


def test_out_of_vocabulary_input_yields_low_confidence(client):
    """
    Text with no known tokens should NOT come back highly confident.

    This is the behaviour the M5 confidence drift signal depends on: if the
    model were confident on meaningless input, a rising OOV rate would not show
    up as falling confidence and the drift signal would be blind.
    """
    body = client.post(
        "/predict/sentiment",
        json={"text": "zzzqqq wwwxxx yyyvvv uuuttt"},
    ).json()
    assert body["confidence"] < 0.75


# ---------------------------------------------------------------------------
# Batch endpoint
# ---------------------------------------------------------------------------

def test_batch_returns_one_result_per_text(client):
    r = client.post(
        "/predict/batch", json={"texts": ["great film", "awful film", "a film"]}
    )
    assert r.status_code == 200
    assert r.json()["count"] == 3


def test_batch_skips_empty_texts_and_reports_the_count(client):
    """Silent dropping is prohibited -- the response says what was skipped."""
    body = client.post(
        "/predict/batch", json={"texts": ["good", "", "   ", "bad"]}
    ).json()
    assert body["count"] == 2
    assert body["skipped"] == 2


def test_batch_of_only_empty_texts_returns_422(client):
    r = client.post("/predict/batch", json={"texts": ["", "  "]})
    assert r.status_code == 422


def test_empty_batch_returns_422(client):
    assert client.post("/predict/batch", json={"texts": []}).status_code == 422


def test_oversized_batch_returns_422(client):
    r = client.post("/predict/batch", json={"texts": ["ok"] * 500})
    assert r.status_code == 422


def test_batch_is_cheaper_per_item_than_single_calls(client):
    """
    Amortisation claim, asserted rather than assumed.

    Sparse matrix construction and the matrix multiply both amortise across a
    batch, so per-item latency should fall well below single-call latency.
    """
    single = client.post("/predict/sentiment", json={"text": "a decent film"}).json()
    batch = client.post(
        "/predict/batch", json={"texts": ["a decent film"] * 50}
    ).json()
    assert batch["per_item_latency_ms"] < single["latency_ms"]


# ---------------------------------------------------------------------------
# Observability
# ---------------------------------------------------------------------------

def test_predictions_are_logged(client):
    """A prediction that was never logged cannot be audited or used for drift."""
    before = client.get("/metrics").json()["predictions_logged"]
    client.post("/predict/sentiment", json={"text": "logging check"})
    after = client.get("/metrics").json()["predictions_logged"]
    assert after > before


def test_metrics_reports_latency_and_confidence(client):
    client.post("/predict/sentiment", json={"text": "metrics check"})
    body = client.get("/metrics").json()
    assert body["avg_latency_ms"] > 0
    assert 0.0 <= body["avg_confidence"] <= 1.0
    assert 0.0 <= body["low_confidence_rate"] <= 1.0
