"""
Drift simulation.

Constructs four traffic windows to exercise the monitor. The important design
point is the FIRST one:

    W0  CONTROL      no drift at all -- must NOT fire
    W1  VOCABULARY   new slang and product names -- OOV should fire
    W2  TOPIC        a different subject domain -- multiple signals should fire
    W3  FORMAT       much longer inputs -- length should fire

--------------------------------------------------------------------------
WHY THE CONTROL WINDOW MATTERS MOST
--------------------------------------------------------------------------
A monitor that always fires is worse than no monitor: it trains the operator to
ignore it, and then the real alert is ignored too. Demonstrating that the
detector STAYS SILENT on undrifted traffic is therefore the primary result, not
a footnote.

W0 is a random resample of the same held-out pool used as the baseline. It is
genuinely undrifted by construction, so any signal that fires on it is a false
positive and a defect in the thresholds.

--------------------------------------------------------------------------
WHY THE DRIFT IS SYNTHETIC, AND WHAT THAT COSTS
--------------------------------------------------------------------------
Real drift would require production traffic collected over months. Within a
14-day project the drift must be constructed, which is an honest limitation:
synthetic drift is cleaner and more abrupt than the real thing. Real drift is
usually gradual, and a gradual shift is harder to detect than these windows
suggest.

The windows are therefore a test of the DETECTOR's wiring and thresholds, not a
claim about real-world sensitivity. That distinction is recorded in
docs/MONITORING.md.
"""

from __future__ import annotations

import random

SEED = 42

# W1 -- vocabulary drift. Terms that post-date the training corpus. Every one is
# out-of-vocabulary, but the SENTIMENT is still expressible, so a human would
# have no trouble. That gap between human and model is the drift.
NEW_SLANG = [
    "this app is absolutely goated no cap",
    "the vibes here are immaculate fr fr",
    "lowkey mid but the rizz carried it",
    "sheeeesh this update is bussin",
    "npc behaviour from the support team ngl",
    "the glow up on this release is unreal",
    "bruh this is straight up cheugy",
    "delulu take but i stan the redesign",
    "gigachad move by the devs tbh",
    "this ain't it chief sadly",
    "caught in 4k being useless",
    "the devs ate and left no crumbs",
    "mad respect the drip is fire",
    "sus rollout but we move",
    "cooked. absolutely cooked.",
]

# W2 -- topic drift. Financial and infrastructure complaints rather than
# film/product opinion. Vocabulary is largely in-dictionary English but the
# SUBJECT is unlike anything in training, so token mix shifts sharply.
NEW_TOPIC = [
    "the mortgage refinancing rate quoted was not honoured at closing",
    "quarterly dividend distribution was delayed by the custodian",
    "escrow account reconciliation shows a variance of four hundred",
    "the amortisation schedule does not match the disclosure statement",
    "wire transfer was rejected due to an intermediary bank hold",
    "my credit utilisation ratio was reported incorrectly to the bureau",
    "the settlement date fell outside the standard clearing window",
    "collateral valuation came in below the underwriting threshold",
    "interest accrual was calculated on a three sixty day basis",
    "the servicer transferred my loan without adequate notice",
    "premium adjustment was applied retroactively to the policy",
    "deductible reset did not occur at the plan anniversary",
    "the vesting schedule accelerated on the change of control",
    "custodial fees were netted against the distribution",
    "the counterparty failed to deliver against the repo",
]


def build_windows(baseline_pool: list[str], n: int = 600) -> dict[str, list[str]]:
    """
    Build four windows of roughly equal size.

    n=600 exceeds the 500-sample minimum in the retraining trigger, so a fired
    trigger is not held back for insufficient data. The HOLD path is exercised
    separately in build_small_window below.
    """
    rng = random.Random(SEED)
    pool = list(baseline_pool)

    # ---- W0: control. Undrifted by construction. -------------------------
    control = rng.sample(pool, min(n, len(pool)))

    # ---- W1: vocabulary drift. 70% new slang, 30% normal traffic. --------
    # Not 100%: real drift arrives mixed with existing traffic, and a monitor
    # that only detects total replacement is not useful.
    n_new = int(n * 0.7)
    vocabulary = [rng.choice(NEW_SLANG) for _ in range(n_new)]
    vocabulary += rng.sample(pool, n - n_new)
    rng.shuffle(vocabulary)

    # ---- W2: topic drift. 70% new domain. --------------------------------
    n_new = int(n * 0.7)
    topic = [rng.choice(NEW_TOPIC) for _ in range(n_new)]
    topic += rng.sample(pool, n - n_new)
    rng.shuffle(topic)

    # ---- W3: format drift. Same words, much longer inputs. ---------------
    # Simulates a new client integration that concatenates several messages
    # into one request. The vocabulary is unchanged, so ONLY the length signal
    # should fire -- which is the point: it isolates one signal.
    fmt = []
    for _ in range(n):
        fmt.append(" ".join(rng.sample(pool, 5)))

    return {
        "W0_control_no_drift": control,
        "W1_vocabulary_drift": vocabulary,
        "W2_topic_drift": topic,
        "W3_format_drift": fmt,
    }


def build_small_window(baseline_pool: list[str], n: int = 120) -> list[str]:
    """
    A drifted window BELOW the minimum-sample threshold.

    Used to demonstrate the HOLD decision: the drift is real, but retraining on
    120 examples would fit the noise in the drift rather than correct for it.
    Detecting drift and being able to act on it are different problems.
    """
    rng = random.Random(SEED)
    out = [rng.choice(NEW_TOPIC) for _ in range(int(n * 0.7))]
    out += rng.sample(list(baseline_pool), n - len(out))
    rng.shuffle(out)
    return out
