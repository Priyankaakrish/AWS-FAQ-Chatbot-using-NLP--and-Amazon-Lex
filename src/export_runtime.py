"""Export the fitted scikit-learn artifacts to a numpy-only bundle.

Produces a single .npz that `src/runtime.py` can score with numpy alone, which
is what makes a small, fast-cold-start Lambda package possible without Docker.

    python -m src.export_runtime --artifacts artifacts --out artifacts/runtime.npz
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .intent_model import IntentClassifier
from .retriever import FaqRetriever


def _tfidf_block(vec) -> dict:
    return {
        "vocabulary": {str(k): int(v) for k, v in vec.vocabulary_.items()},
        "analyzer": vec.analyzer,
        "min_n": int(vec.ngram_range[0]),
        "max_n": int(vec.ngram_range[1]),
        "sublinear_tf": bool(vec.sublinear_tf),
    }


def export(artifacts_dir: str, out_path: str) -> dict:
    pipeline = IntentClassifier.load(os.path.join(artifacts_dir, "intent_model.joblib")).pipeline
    index = FaqRetriever.load(os.path.join(artifacts_dir, "faq_index.joblib"))

    arrays: dict[str, np.ndarray] = {}

    # ---- intent classifier -------------------------------------------------
    union = pipeline.named_steps["features"]
    blocks = []
    for i, (_, vec) in enumerate(union.transformer_list):
        blocks.append(_tfidf_block(vec))
        arrays[f"clf_idf_{i}"] = vec.idf_.astype(np.float64)

    calibrated = pipeline.named_steps["clf"]
    folds = []
    for i, cc in enumerate(calibrated.calibrated_classifiers_):
        est = cc.estimator
        arrays[f"clf_coef_{i}"] = np.asarray(est.coef_, dtype=np.float64)
        arrays[f"clf_intercept_{i}"] = np.asarray(est.intercept_, dtype=np.float64)
        arrays[f"clf_a_{i}"] = np.array([c.a_ for c in cc.calibrators], dtype=np.float64)
        arrays[f"clf_b_{i}"] = np.array([c.b_ for c in cc.calibrators], dtype=np.float64)
        folds.append({})

    classifier_json = {
        "blocks": blocks,
        "folds": folds,
        "classes": [str(c) for c in calibrated.classes_],
    }

    # ---- retriever ---------------------------------------------------------
    backend = index.backend
    steps = backend.model.steps
    vec = steps[0][1]
    svd = steps[1][1]
    arrays["ret_idf"] = vec.idf_.astype(np.float64)
    arrays["ret_components"] = np.asarray(svd.components_, dtype=np.float64)
    arrays["ret_matrix"] = np.asarray(index.matrix, dtype=np.float64)

    retriever_json = {
        "tfidf": _tfidf_block(vec),
        "meta": index.meta,
        "vocab_tokens": sorted(index.vocab),
        "threshold": float(index.threshold),
        "retry_threshold": float(getattr(index, "retry_threshold", index.threshold)),
        "coverage_threshold": float(index.coverage_threshold),
        "intent_floor": float(index.intent_floor),
        "stop_words": sorted(ENGLISH_STOP_WORDS),
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        classifier_json=np.array(json.dumps(classifier_json)),
        retriever_json=np.array(json.dumps(retriever_json)),
        **arrays,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    return {
        "path": out_path,
        "size_mb": round(size_mb, 2),
        "n_classes": len(classifier_json["classes"]),
        "n_folds": len(folds),
        "n_faqs": len(index.meta),
        "feature_dims": [len(b["vocabulary"]) for b in blocks],
        "svd_components": int(arrays["ret_components"].shape[0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="artifacts/runtime.npz")
    args = ap.parse_args()
    info = export(args.artifacts, args.out)
    print(json.dumps(info, indent=2))
    print(f"\nLambda now needs numpy only — no scikit-learn, no scipy, no Docker.")


if __name__ == "__main__":
    main()
