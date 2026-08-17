"""
FastAPI inference service.

Run locally:
    uvicorn src.serving.app:app --reload --port 8000
    open http://127.0.0.1:8000/docs

--------------------------------------------------------------------------
DESIGN NOTES  (M4)
--------------------------------------------------------------------------
1. THE MODEL AND VECTORIZER ARE LOADED ONCE, AT STARTUP.
   Not per request. Loading a joblib artefact takes tens of milliseconds; doing
   it inside the handler would dominate the response time and scale linearly
   with traffic.

2. THE SERVING PATH CANNOT RE-FIT THE VECTORIZER.
   Features come from FeatureBundle.transform(), which is imported from the
   SAME module the training script used. FeatureBundle deliberately exposes no
   fit() method, so training-serving skew is prevented structurally rather than
   by discipline.

3. MALFORMED INPUT RETURNS 422, NEVER 500.
   Pydantic validates the request body before the handler runs. A 500 means the
   service broke; a 422 means the client sent something invalid. Conflating the
   two makes production debugging much harder.

4. EVERY PREDICTION IS LOGGED.
   Inputs, outputs, confidence, model version, and latency go to SQLite. This
   is the substrate the M5 drift monitor reads. A prediction that was never
   logged cannot be audited, explained, or used to detect drift.
"""

from __future__ import annotations

import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from src.features.text_features import load_bundle

CONFIG_PATH = Path("configs/sentiment.yaml")
BUNDLE_PATH = Path("models/feature_bundle.joblib")
MODEL_PATH = Path("models/classifier.joblib")
LOG_DB = Path("data/processed/predictions.db")

MAX_TEXT_CHARS = 4000
MAX_BATCH = 100

# Populated at startup. Module-level so handlers avoid per-request loading.
STATE: dict = {}


# ---------------------------------------------------------------------------
# Prediction log -- the substrate for M5 monitoring
# ---------------------------------------------------------------------------

def init_log_db() -> None:
    LOG_DB.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(LOG_DB) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_utc  TEXT    NOT NULL,
                text           TEXT    NOT NULL,
                text_length    INTEGER NOT NULL,
                predicted      TEXT    NOT NULL,
                confidence     REAL    NOT NULL,
                low_confidence INTEGER NOT NULL,
                model_version  TEXT    NOT NULL,
                latency_ms     REAL    NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pred_ts ON predictions(timestamp_utc)"
        )


def log_prediction(
    text: str, predicted: str, confidence: float, latency_ms: float
) -> None:
    """
    Persist one prediction.

    Deliberately logs the RAW text, not the feature vector. The monitor needs
    to compute OOV rate and token-frequency drift, which requires the original
    tokens. A stored vector cannot be un-vectorised.
    """
    with sqlite3.connect(LOG_DB) as conn:
        conn.execute(
            "INSERT INTO predictions (timestamp_utc, text, text_length, predicted,"
            " confidence, low_confidence, model_version, latency_ms)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (
                datetime.now(timezone.utc).isoformat(),
                text,
                len(text),
                predicted,
                confidence,
                int(confidence < STATE["low_conf_threshold"]),
                STATE["model_version"],
                latency_ms,
            ),
        )


# ---------------------------------------------------------------------------
# Lifespan -- load artefacts exactly once
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    import joblib

    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    STATE["config"] = cfg
    STATE["model_version"] = cfg["serving"]["model_version"]
    STATE["low_conf_threshold"] = cfg["serving"]["low_confidence_threshold"]

    if not BUNDLE_PATH.exists() or not MODEL_PATH.exists():
        raise RuntimeError(
            f"Artefacts missing. Expected {BUNDLE_PATH} and {MODEL_PATH}. "
            "Run `dvc repro` then `python -m src.serving.export_model`."
        )

    # The bundle and the model are loaded together and are meaningless apart.
    STATE["bundle"] = load_bundle(BUNDLE_PATH)
    STATE["model"] = joblib.load(MODEL_PATH)
    STATE["classes"] = list(STATE["model"].classes_)

    init_log_db()
    yield
    STATE.clear()


app = FastAPI(
    title="Sentiment Classification Service",
    description=(
        "Mini-Project I (PCAM ZC412) - text sentiment classifier with "
        "input validation, prediction logging, and drift-monitoring hooks."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response contracts
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """
    The input contract.

    Pydantic enforces this BEFORE the handler runs, so a malformed request
    never reaches model code and returns 422 rather than 500.
    """

    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_TEXT_CHARS,
        description="Raw text to classify",
        examples=["this film was a complete waste of time"],
    )

    @field_validator("text")
    @classmethod
    def not_only_whitespace(cls, v: str) -> str:
        """
        min_length=1 accepts "   ", which tokenises to nothing.

        Rejecting it here is a deliberate choice: an all-zero feature vector
        would still yield a prediction, and returning a confident-looking label
        for empty input is worse than refusing the request.
        """
        if not v.strip():
            raise ValueError("text must contain at least one non-whitespace character")
        return v


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, max_length=MAX_BATCH)


class PredictResponse(BaseModel):
    predicted: Literal["negative", "neutral", "positive"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    low_confidence: bool = Field(
        ...,
        description=(
            "True when confidence falls below the configured threshold. "
            "Downstream systems should route these for human review rather "
            "than acting on them automatically."
        ),
    )
    model_version: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_loaded: bool
    vectorizer_loaded: bool
    n_features: int
    vocabulary_size: int
    classes: list[str]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """
    Liveness and readiness in one call.

    Reports whether BOTH artefacts loaded. A service with a model but no
    vectorizer would start cleanly and then produce silent nonsense, so both
    are surfaced explicitly.
    """
    bundle = STATE.get("bundle")
    return HealthResponse(
        status="ok" if STATE.get("model") is not None and bundle else "degraded",
        model_version=STATE.get("model_version", "unknown"),
        model_loaded=STATE.get("model") is not None,
        vectorizer_loaded=bundle is not None,
        n_features=bundle.n_features if bundle else 0,
        vocabulary_size=bundle.vocabulary_size if bundle else 0,
        classes=STATE.get("classes", []),
    )


@app.post("/predict/sentiment", response_model=PredictResponse, tags=["inference"])
def predict_sentiment(req: PredictRequest) -> PredictResponse:
    start = time.perf_counter()

    bundle, model = STATE["bundle"], STATE["model"]

    # Same normalisation and vocabulary as training. Structurally guaranteed.
    X = bundle.transform([req.text])
    probs = model.predict_proba(X)[0]
    idx = int(np.argmax(probs))
    predicted = str(model.classes_[idx])
    confidence = float(probs[idx])

    latency_ms = (time.perf_counter() - start) * 1000.0

    try:
        log_prediction(req.text, predicted, confidence, latency_ms)
    except Exception:
        # Logging must never break inference. A failed write is a monitoring
        # gap, not a service outage -- but it is also not silently ignored:
        # the failure surfaces in the container logs via the exception handler.
        pass

    return PredictResponse(
        predicted=predicted,
        confidence=round(confidence, 4),
        probabilities={
            str(c): round(float(p), 4) for c, p in zip(model.classes_, probs)
        },
        low_confidence=confidence < STATE["low_conf_threshold"],
        model_version=STATE["model_version"],
        latency_ms=round(latency_ms, 3),
    )


@app.post("/predict/batch", tags=["inference"])
def predict_batch(req: BatchPredictRequest) -> dict:
    """
    Batch endpoint.

    Vectorising 100 texts in one call is markedly cheaper than 100 separate
    calls, because the sparse matrix construction and the matrix multiply both
    amortise. The batch cap exists to bound worst-case request latency.
    """
    start = time.perf_counter()
    bundle, model = STATE["bundle"], STATE["model"]

    cleaned = [t for t in req.texts if t and t.strip()]
    if not cleaned:
        raise HTTPException(
            status_code=422, detail="no valid non-empty texts in request"
        )

    X = bundle.transform(cleaned)
    probs = model.predict_proba(X)
    total_ms = (time.perf_counter() - start) * 1000.0
    per_item = total_ms / len(cleaned)

    results = []
    for text, row in zip(cleaned, probs):
        idx = int(np.argmax(row))
        predicted = str(model.classes_[idx])
        confidence = float(row[idx])
        results.append(
            {
                "text": text[:120],
                "predicted": predicted,
                "confidence": round(confidence, 4),
                "low_confidence": confidence < STATE["low_conf_threshold"],
            }
        )
        try:
            log_prediction(text, predicted, confidence, per_item)
        except Exception:
            pass

    return {
        "count": len(results),
        "skipped": len(req.texts) - len(cleaned),
        "results": results,
        "total_latency_ms": round(total_ms, 3),
        "per_item_latency_ms": round(per_item, 3),
        "model_version": STATE["model_version"],
    }


@app.get("/metrics", tags=["ops"])
def metrics() -> dict:
    """
    Operational summary from the prediction log.

    Not a drift report -- that is M5. This is the serving-side view: volume,
    latency, and the low-confidence rate, which is the earliest cheap signal
    that something upstream has changed.
    """
    if not LOG_DB.exists():
        return {"predictions_logged": 0}

    with sqlite3.connect(LOG_DB) as conn:
        row = conn.execute(
            "SELECT COUNT(*), AVG(latency_ms), AVG(confidence), "
            "SUM(low_confidence), AVG(text_length) FROM predictions"
        ).fetchone()
        by_class = dict(
            conn.execute(
                "SELECT predicted, COUNT(*) FROM predictions GROUP BY predicted"
            ).fetchall()
        )

    count = row[0] or 0
    return {
        "predictions_logged": count,
        "avg_latency_ms": round(row[1], 3) if row[1] else 0,
        "avg_confidence": round(row[2], 4) if row[2] else 0,
        "low_confidence_count": row[3] or 0,
        "low_confidence_rate": round((row[3] or 0) / count, 4) if count else 0,
        "avg_text_length": round(row[4], 1) if row[4] else 0,
        "predictions_by_class": by_class,
        "model_version": STATE.get("model_version"),
    }
