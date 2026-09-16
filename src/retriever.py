"""Semantic FAQ retrieval.

Three interchangeable embedding backends so the same evaluation harness can
compare them:

  lsa      TF-IDF + TruncatedSVD. No downloads, no GPU, ~2 MB artifact — this is
           the one that actually fits in a Lambda zip.
  sbert    sentence-transformers (all-MiniLM-L6-v2). Best offline quality.
  bedrock  Amazon Titan Text Embeddings v2 via bedrock-runtime. No model to
           ship, but adds a network hop per query.

All backends return L2-normalised vectors so cosine similarity is a dot
product and scores are comparable across backends.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol

import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .artifacts import safe_load
from .preprocess import normalize

STOP_WORDS = set(ENGLISH_STOP_WORDS)


class EmbeddingBackend(Protocol):
    def fit(self, corpus: list[str]) -> "EmbeddingBackend": ...
    def encode(self, texts: list[str]) -> np.ndarray: ...


class LsaBackend:
    """TF-IDF -> SVD -> L2 normalise. Trainable, tiny, deterministic."""

    name = "lsa"

    def __init__(self, n_components: int = 256, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.model = None

    def fit(self, corpus: list[str]) -> "LsaBackend":
        n = min(self.n_components, max(2, len(corpus) - 1))
        self.model = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True),
            TruncatedSVD(n_components=n, random_state=self.random_state),
            Normalizer(copy=False),
        )
        self.model.fit([normalize(c) for c in corpus])
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("LsaBackend.fit must be called before encode")
        return np.asarray(self.model.transform([normalize(t) for t in texts]))


class SbertBackend:
    name = "sbert"

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import

        self.model = SentenceTransformer(model_name)

    def fit(self, corpus: list[str]) -> "SbertBackend":
        return self  # pretrained

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        )


class BedrockBackend:
    name = "bedrock"

    def __init__(self, model_id: str = "amazon.titan-embed-text-v2:0", region: str | None = None):
        import boto3  # lazy import

        self.model_id = model_id
        self.client = boto3.client("bedrock-runtime",
                                   region_name=region or os.environ.get("AWS_REGION", "us-east-1"))

    def fit(self, corpus: list[str]) -> "BedrockBackend":
        return self

    def encode(self, texts: list[str]) -> np.ndarray:
        import json

        vectors = []
        for t in texts:
            resp = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({"inputText": t, "dimensions": 512, "normalize": True}),
            )
            vectors.append(json.loads(resp["body"].read())["embedding"])
        return np.asarray(vectors, dtype=np.float32)


def get_backend(name: str, **kwargs) -> EmbeddingBackend:
    return {"lsa": LsaBackend, "sbert": SbertBackend, "bedrock": BedrockBackend}[name](**kwargs)


@dataclass
class RetrievalHit:
    faq_id: str
    intent: str
    response: str
    score: float
    rank: int


class FaqRetriever:
    """Dense index over FAQ entries with cosine scoring and fallback logic."""

    def __init__(self, backend: EmbeddingBackend, threshold: float = 0.45,
                 coverage_threshold: float = 0.30, intent_floor: float = 0.40,
                 retry_threshold: float = 0.45):
        self.backend = backend
        self.threshold = threshold
        # Separate knob for "abandon the intent-filtered search and re-search
        # globally". It used to share `threshold`, so raising the fallback bar
        # also made the pipeline discard correct intent-filtered hits in favour
        # of whole-index matches from the wrong intent -- measured as a 4-point
        # top-1 accuracy *drop* when threshold went 0.45 -> 0.70. One constant,
        # two unrelated responsibilities.
        self.retry_threshold = retry_threshold
        self.coverage_threshold = coverage_threshold
        self.intent_floor = intent_floor
        self.matrix: np.ndarray | None = None
        self.meta: list[dict] = []
        self.vocab: set[str] = set()

    def fit(self, index_texts: list[str], meta: list[dict],
            fit_backend: bool = True) -> "FaqRetriever":
        if fit_backend:
            self.backend.fit(index_texts)
        self.matrix = self.backend.encode(index_texts)
        self.meta = meta
        # Backend-agnostic lexical vocabulary, used for the OOV fallback gate.
        self.vocab = {tok for text in index_texts for tok in normalize(text).split()}
        return self

    def lexical_coverage(self, query: str) -> float:
        """Fraction of the query's content words that the index has ever seen.

        Dense scores are not trustworthy on their own: a low-rank projection
        maps genuinely unknown text onto whatever direction is closest, which
        can look like high cosine similarity. Vocabulary overlap is a cheap,
        orthogonal signal that catches exactly that failure.
        """
        tokens = [t for t in normalize(query).split() if t not in STOP_WORDS and len(t) > 2]
        if not tokens:
            return 0.0
        return sum(t in self.vocab for t in tokens) / len(tokens)

    def search(self, query: str, k: int = 5,
               intent_filter: str | None = None) -> list[RetrievalHit]:
        if self.matrix is None:
            raise RuntimeError("FaqRetriever.fit must be called before search")
        q = self.backend.encode([query])[0]
        scores = self.matrix @ q
        if intent_filter:
            mask = np.array([m["intent"] == intent_filter for m in self.meta])
            if mask.any():
                scores = np.where(mask, scores, -np.inf)
        order = np.argsort(scores)[::-1][:k]
        return [
            RetrievalHit(
                faq_id=self.meta[i]["faq_id"],
                intent=self.meta[i]["intent"],
                response=self.meta[i]["response"],
                score=float(scores[i]),
                rank=rank,
            )
            for rank, i in enumerate(order, start=1)
            if np.isfinite(scores[i])
        ]

    def fallback_reason(self, hits: list[RetrievalHit], query: str,
                        intent_confidence: float | None = None) -> str | None:
        """Return why we should refuse to answer, or None to answer.

        Three independent gates, cheapest first. Returning the *reason* rather
        than a bool matters operationally: 'out_of_vocabulary' means the KB has
        a content gap worth filling, while 'ambiguous_intent' means two intents
        need better training utterances. Those are different tickets.
        """
        if not hits:
            return "no_candidates"
        if self.lexical_coverage(query) < self.coverage_threshold:
            return "out_of_vocabulary"
        if hits[0].score < self.threshold:
            return "low_similarity"
        if (
            intent_confidence is not None
            and intent_confidence < self.intent_floor
            and hits[0].score < self.threshold + 0.15
        ):
            return "ambiguous_intent"
        return None

    def is_fallback(self, hits: list[RetrievalHit], query: str,
                    intent_confidence: float | None = None) -> bool:
        return self.fallback_reason(hits, query, intent_confidence) is not None

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump(
            {
                "backend": self.backend if isinstance(self.backend, LsaBackend) else None,
                "backend_name": getattr(self.backend, "name", "lsa"),
                "matrix": self.matrix,
                "meta": self.meta,
                "vocab": self.vocab,
                "threshold": self.threshold,
                "retry_threshold": self.retry_threshold,
                "coverage_threshold": self.coverage_threshold,
                "intent_floor": self.intent_floor,
            },
            path,
        )

    @classmethod
    def load(cls, path: str, backend: EmbeddingBackend | None = None) -> "FaqRetriever":
        blob = safe_load(path)
        backend = backend or blob["backend"] or get_backend(blob["backend_name"])
        obj = cls(backend, blob["threshold"], blob["coverage_threshold"], blob["intent_floor"],
                  blob.get("retry_threshold", blob["threshold"]))
        obj.matrix = blob["matrix"]
        obj.meta = blob["meta"]
        obj.vocab = blob.get("vocab", set())
        return obj
