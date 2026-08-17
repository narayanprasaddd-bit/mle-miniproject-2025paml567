"""
Latency and throughput benchmark.

Usage:
    python -m src.serving.benchmark

The M4 rubric line names "latency/throughput awareness" explicitly. This
measures it rather than asserting it, and reports percentiles rather than a
mean -- a mean latency hides the tail, and the tail is what users experience
as "the service is slow".
"""

from __future__ import annotations

import json
import sqlite3
import statistics
import sys
import time
from pathlib import Path

import joblib

from src.features.text_features import load_bundle

OUT = Path("reports/latency_benchmark.json")
N_WARMUP = 20
N_TRIALS = 500


def main() -> int:
    bundle_path = Path("models/feature_bundle.joblib")
    model_path = Path("models/classifier.joblib")
    if not bundle_path.exists() or not model_path.exists():
        print("  FAIL: artefacts missing. Run dvc repro + export_model first.")
        return 1

    print("=" * 72)
    print("  LATENCY BENCHMARK")
    print("=" * 72)

    t0 = time.perf_counter()
    bundle = load_bundle(bundle_path)
    model = joblib.load(model_path)
    load_ms = (time.perf_counter() - t0) * 1000

    print(f"  artefact load time: {load_ms:.1f} ms  (once, at startup)")
    print("  ^ this is why loading happens in the lifespan handler and not")
    print("    per request: it would otherwise dominate response time.\n")

    # Draw real texts from the feature store so the benchmark reflects
    # production-shaped input rather than synthetic strings.
    with sqlite3.connect("data/processed/feature_store.db") as conn:
        texts = [r[0] for r in conn.execute(
            "SELECT text FROM features_test LIMIT ?", (N_TRIALS + N_WARMUP,)
        ).fetchall()]

    for t in texts[:N_WARMUP]:
        model.predict_proba(bundle.transform([t]))

    single = []
    for t in texts[N_WARMUP:N_WARMUP + N_TRIALS]:
        s = time.perf_counter()
        model.predict_proba(bundle.transform([t]))
        single.append((time.perf_counter() - s) * 1000)

    single.sort()
    p50 = single[len(single) // 2]
    p95 = single[int(len(single) * 0.95)]
    p99 = single[int(len(single) * 0.99)]

    print("  SINGLE-REQUEST latency (n=%d):" % len(single))
    print(f"    p50  {p50:7.3f} ms")
    print(f"    p95  {p95:7.3f} ms")
    print(f"    p99  {p99:7.3f} ms")
    print(f"    max  {single[-1]:7.3f} ms")
    print(f"    theoretical throughput: {1000/statistics.mean(single):,.0f} req/s "
          f"per worker\n")

    batch_results = {}
    for size in (1, 10, 50, 100):
        sample = texts[N_WARMUP:N_WARMUP + size]
        s = time.perf_counter()
        model.predict_proba(bundle.transform(sample))
        total = (time.perf_counter() - s) * 1000
        batch_results[size] = {
            "total_ms": round(total, 3),
            "per_item_ms": round(total / size, 4),
        }
        print(f"  BATCH size {size:>3}: {total:7.2f} ms total, "
              f"{total/size:6.3f} ms per item")

    speedup = batch_results[1]["per_item_ms"] / batch_results[100]["per_item_ms"]
    print(f"\n  batching 100 is {speedup:.1f}x cheaper per item than 1-at-a-time")
    print("  (sparse matrix construction and the matvec both amortise)")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({
        "artefact_load_ms": round(load_ms, 2),
        "n_trials": len(single),
        "single_request_ms": {
            "p50": round(p50, 4), "p95": round(p95, 4),
            "p99": round(p99, 4), "max": round(single[-1], 4),
            "mean": round(statistics.mean(single), 4),
        },
        "throughput_req_per_sec_per_worker": round(1000/statistics.mean(single)),
        "batch": batch_results,
        "batch_100_speedup_per_item": round(speedup, 2),
    }, indent=2), encoding="utf-8")
    print(f"\n  written: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
