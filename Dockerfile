# =============================================================================
# Sentiment classification service - production image
# =============================================================================
# Multi-stage build. The builder compiles wheels; the runtime carries only what
# is needed to serve. This matters for three reasons:
#
#   1. SIZE      - build toolchains (gcc, headers) are large and unnecessary at
#                  runtime, so they never reach the final image.
#   2. SECURITY  - a smaller image has a smaller attack surface. No compiler
#                  means no compiler vulnerabilities to patch.
#   3. STARTUP   - fewer layers and files means faster cold starts.
# =============================================================================

# ----------------------------- STAGE 1: build --------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Only the serving dependencies. mlflow, dvc, and pytest are development
# tooling and are deliberately absent from the runtime image.
RUN pip install --no-cache-dir --prefix=/install \
        fastapi==0.111.* \
        "uvicorn[standard]==0.30.*" \
        pydantic==2.8.* \
        scikit-learn==1.5.* \
        pandas==2.2.* \
        numpy==1.26.* \
        joblib==1.4.* \
        pyyaml==6.0.*

# ---------------------------- STAGE 2: runtime -------------------------------
FROM python:3.11-slim

# Run as a non-root user. A container process that does not need root should
# not have it: if the application is compromised, the attacker inherits an
# unprivileged account rather than the container's root.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy only what serving needs. Ordered least- to most-frequently-changed so
# Docker's layer cache is not invalidated by an unrelated edit.
COPY configs/sentiment.yaml       configs/sentiment.yaml
COPY src/__init__.py              src/__init__.py
COPY src/features/__init__.py     src/features/__init__.py
COPY src/features/text_features.py src/features/text_features.py
COPY src/serving/__init__.py      src/serving/__init__.py
COPY src/serving/app.py           src/serving/app.py

# The model artefacts. BOTH are required - the classifier is meaningless
# without the vectorizer that produced its feature space.
COPY models/classifier.joblib     models/classifier.joblib
COPY models/feature_bundle.joblib models/feature_bundle.joblib

# The prediction log is written at runtime, so its directory must be writable
# by the non-root user.
RUN mkdir -p data/processed && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# Container-native health check. An orchestrator uses this to decide whether
# the container is ready for traffic and whether to restart it.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

# Single worker by design. The model is loaded into each worker's memory, so
# N workers means N copies of the artefacts. Scale by running more CONTAINERS
# rather than more workers in one container - that keeps memory predictable
# and lets the orchestrator do the scheduling.
CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
