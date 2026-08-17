"""
THE SHARED FEATURE MODULE.

This is the most important file in the project.

Both the training script (src/training/) and the serving application
(src/serving/) import their feature logic from HERE. Neither implements its own.
That is the entire point: if training and serving normalise text differently,
tokenise differently, or use different vocabularies, the model receives inputs
at serving time that do not resemble what it was trained on, and every
prediction shifts -- silently, with the API returning HTTP 200 throughout.

M2 2.6 calls this TRAINING-SERVING SKEW. The prescribed remedy is a single
shared transformation path, which is what this module is.

--------------------------------------------------------------------------
THE TRAINING-SERVING CONTRACT
--------------------------------------------------------------------------
In the instructor's tabular tutorial, the contract is feature_schema.json --
a list of column names. In an NLP system the contract is much heavier: it is
the FITTED VECTORIZER itself, carrying its vocabulary and its IDF weights.

    A TF-IDF vector is meaningless without the vocabulary that produced it.
    Token index 4,182 means "disappointing" only in the vocabulary that
    assigned it. Re-fit the vectorizer on different text and index 4,182
    becomes some other word -- and the model's learned coefficient for that
    position now applies to the wrong feature.

Therefore the vectorizer MUST be:
    - fitted exactly once, on training data only
    - persisted alongside the model as one inseparable bundle
    - loaded once at API startup, never re-fitted
    - versioned in DVC and logged as an MLflow artifact

The artefact is not "the model". It is "the model AND the vectorizer",
frozen together, because either alone is meaningless to the other.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Text normalisation -- the ONE place this logic exists
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """
    Canonical text normalisation. Called by training AND serving.

    Deliberately minimal. Every transformation here is a transformation the
    serving path must perform identically, so each one is a liability. Only
    steps justified by an evidenced finding are included:

      - str() coercion : serving receives JSON, which may deliver a non-string
      - .strip()       : BR-03, 39.9% of training rows had leading whitespace

    Lowercasing is NOT done here. It is delegated to the vectorizer via
    lowercase=True, so that the setting travels inside the pickled artefact
    rather than living in code that could drift out of sync with it.
    """
    return str(text).strip()


# ---------------------------------------------------------------------------
# The bundle -- model artefacts that must never be separated
# ---------------------------------------------------------------------------

@dataclass
class FeatureBundle:
    """
    A fitted vectorizer plus the metadata needed to audit it.

    Persisted as a single file. Loading a model without its bundle is an error
    the type system cannot prevent, so the two are stored together and the
    loader returns them as a unit.
    """

    vectorizer: TfidfVectorizer
    config_fingerprint: dict
    n_features: int
    vocabulary_size: int

    def transform(self, texts: list[str]):
        """
        Transform raw text into the model's feature space.

        This is the method serving calls. It normalises first, then vectorises
        with the ALREADY-FITTED vectorizer. There is no fit() here by design --
        the class does not expose a way to accidentally re-fit at serving time.
        """
        normalised = [normalise_text(t) for t in texts]
        return self.vectorizer.transform(normalised)

    def oov_rate(self, texts: list[str]) -> float:
        """
        Share of UNIGRAM tokens absent from the fitted vocabulary.

        Used by the M5 drift monitor. A rising OOV rate is the cleanest signal
        that production language has moved away from the training corpus --
        new slang, new product names, a new topic.

        ------------------------------------------------------------------
        WHY UNIGRAMS ONLY -- an empirical finding, not a preference
        ------------------------------------------------------------------
        Measured on the held-out test split of this corpus:

            ngram_range=(1,2), max_features=20,000  ->  OOV = 0.3191
            ngram_range=(1,1), max_features=20,000  ->  OOV = 0.0775
            ngram_range=(1,2), max_features=50,000  ->  OOV = 0.2909

        Raising the feature cap from 20k to 50k barely moved the figure, which
        rules out the cap as the cause. Bigrams are inherently sparse: most
        word PAIRS in any unseen text have genuinely never occurred in
        training, even when every individual word is familiar.

        That makes bigram OOV a poor drift signal, because it is already near
        saturation before any drift occurs. A signal sitting at 32% with little
        headroom cannot cleanly indicate a change; the drift would be lost in
        the noise floor.

        Unigram OOV at 7.75% has room to move and responds to the thing we
        actually care about -- new WORDS entering the traffic. The 0.15
        threshold in configs/*.yaml is set against this unigram baseline,
        roughly double it.

        The bigrams remain in the model's feature space; they are useful for
        prediction. They are simply excluded from this measurement.
        """
        analyzer = self.vectorizer.build_analyzer()
        # sklearn joins n-gram components with a single space, so a token
        # containing no space is a unigram.
        unigram_vocab = {t for t in self.vectorizer.vocabulary_ if " " not in t}
        total = oov = 0
        for t in texts:
            for tok in analyzer(normalise_text(t)):
                if " " in tok:
                    continue
                total += 1
                if tok not in unigram_vocab:
                    oov += 1
        return (oov / total) if total else 0.0


# ---------------------------------------------------------------------------
# Fitting -- called ONLY by training, never by serving
# ---------------------------------------------------------------------------

def fit_vectorizer(texts: list[str], features_cfg: dict) -> FeatureBundle:
    """
    Fit the TF-IDF vectorizer on training text only.

    Called exactly once per model version, from the training script. The
    resulting bundle is then frozen for the lifetime of that model.
    """
    vec = TfidfVectorizer(
        lowercase=features_cfg["lowercase"],
        ngram_range=tuple(features_cfg["ngram_range"]),
        max_features=features_cfg["max_features"],
        min_df=features_cfg["min_df"],
        sublinear_tf=features_cfg["sublinear_tf"],
        stop_words=features_cfg.get("stop_words"),
    )
    normalised = [normalise_text(t) for t in texts]
    matrix = vec.fit_transform(normalised)

    return FeatureBundle(
        vectorizer=vec,
        config_fingerprint={
            "lowercase": features_cfg["lowercase"],
            "ngram_range": list(features_cfg["ngram_range"]),
            "max_features": features_cfg["max_features"],
            "min_df": features_cfg["min_df"],
            "sublinear_tf": features_cfg["sublinear_tf"],
            "stop_words": features_cfg.get("stop_words"),
            "normalisation": "str().strip()",
        },
        n_features=matrix.shape[1],
        vocabulary_size=len(vec.vocabulary_),
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_bundle(bundle: FeatureBundle, path: str | Path) -> None:
    """Persist the bundle, plus a human-readable sidecar for auditing."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)

    sidecar = path.with_suffix(".json")
    with open(sidecar, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "artefact": str(path.name),
                "n_features": bundle.n_features,
                "vocabulary_size": bundle.vocabulary_size,
                "config_fingerprint": bundle.config_fingerprint,
                "contract_note": (
                    "This vectorizer IS the training-serving contract. The model "
                    "trained against it is invalid with any other vectorizer."
                ),
            },
            fh,
            indent=2,
        )


def load_bundle(path: str | Path) -> FeatureBundle:
    """Load a fitted bundle. Used by serving at startup, exactly once."""
    return joblib.load(path)
