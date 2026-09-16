"""Intent classification model.

Baseline that is fast enough to run inside a Lambda cold start: a union of word
and character n-gram TF-IDF features feeding a calibrated linear SVM. The
calibration wrapper is what gives us a usable confidence score for fallback
routing — an uncalibrated SVM decision function is not comparable across
intents.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from .artifacts import safe_load
from .preprocess import normalize


@dataclass
class IntentPrediction:
    intent: str
    confidence: float
    margin: float           # top1 - top2 probability
    top_k: list[tuple[str, float]]


def build_pipeline(calibrate: bool = True, cv: int = 3) -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                     min_df=3, sublinear_tf=True)),
        ]
    )
    base = LinearSVC(C=1.0, class_weight="balanced")
    clf = CalibratedClassifierCV(base, cv=cv, method="sigmoid") if calibrate else base
    return Pipeline([("features", features), ("clf", clf)])


class IntentClassifier:
    def __init__(self, pipeline: Pipeline | None = None):
        self.pipeline = pipeline or build_pipeline()

    def fit(self, texts, labels) -> "IntentClassifier":
        self.pipeline.fit([normalize(t) for t in texts], list(labels))
        return self

    @property
    def classes_(self) -> np.ndarray:
        return self.pipeline.named_steps["clf"].classes_

    def predict(self, texts) -> list[str]:
        return list(self.pipeline.predict([normalize(t) for t in texts]))

    def predict_proba(self, texts) -> np.ndarray:
        return self.pipeline.predict_proba([normalize(t) for t in texts])

    def predict_one(self, text: str, k: int = 3) -> IntentPrediction:
        proba = self.predict_proba([text])[0]
        order = np.argsort(proba)[::-1]
        classes = self.classes_
        top = [(str(classes[i]), float(proba[i])) for i in order[:k]]
        margin = float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0
        return IntentPrediction(top[0][0], top[0][1], margin, top)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path: str) -> "IntentClassifier":
        return cls(safe_load(path))
