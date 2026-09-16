"""Dependency-free inference runtime.

scikit-learn, scipy and numpy together are roughly 150 MB unzipped. That is
awkward inside Lambda's 250 MB limit and makes cold starts slow, and the usual
workaround — a container image — drags Docker into the build.

But nothing on the *inference* path actually needs scikit-learn. Scoring a
query is TF-IDF, a linear decision function, three sigmoids and a matrix
product. This module reimplements exactly that against plain numpy arrays
exported by `export_runtime.py`, so the Lambda package needs numpy alone.

Equivalence with the scikit-learn pipeline is asserted numerically in
`tests/test_runtime.py`; if sklearn's internals change, that test fails rather
than the model silently drifting.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

import numpy as np

TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")
MULTI_WS = re.compile(r"\s\s+")


# ------------------------------------------------------ feature extraction ---
def word_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    """Mirror of sklearn's word analyzer (lowercase + token_pattern + n-grams)."""
    tokens = TOKEN_RE.findall(text.lower())
    if max_n == 1:
        return tokens
    out = list(tokens) if min_n == 1 else []
    n_tokens = len(tokens)
    for n in range(max(min_n, 2), max_n + 1):
        for i in range(n_tokens - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


def char_wb_ngrams(text: str, min_n: int, max_n: int) -> list[str]:
    """Mirror of sklearn's `char_wb` analyzer.

    Each whitespace-delimited word is padded with a single space on both sides,
    then character n-grams are taken inside that padded word only — so n-grams
    never span a word boundary. Short words yield the padded word once, which is
    the `if offset == 0: break` case in sklearn.
    """
    text = MULTI_WS.sub(" ", text.lower())
    out: list[str] = []
    for w in text.split():
        w = " " + w + " "
        w_len = len(w)
        for n in range(min_n, max_n + 1):
            offset = 0
            out.append(w[offset : offset + n])
            while offset + n < w_len:
                offset += 1
                out.append(w[offset : offset + n])
            if offset == 0:
                break
    return out


@dataclass
class TfidfBlock:
    """One exported TfidfVectorizer: vocabulary, idf weights, analyzer config."""

    vocabulary: dict[str, int]
    idf: np.ndarray
    analyzer: str
    min_n: int
    max_n: int
    sublinear_tf: bool = True

    def transform(self, text: str) -> np.ndarray:
        features = (
            word_ngrams(text, self.min_n, self.max_n)
            if self.analyzer == "word"
            else char_wb_ngrams(text, self.min_n, self.max_n)
        )
        vec = np.zeros(len(self.idf), dtype=np.float64)
        for f in features:
            idx = self.vocabulary.get(f)
            if idx is not None:
                vec[idx] += 1.0
        if self.sublinear_tf:
            nz = vec > 0
            vec[nz] = 1.0 + np.log(vec[nz])
        vec *= self.idf
        norm = np.linalg.norm(vec)
        # Each sub-vectorizer L2-normalises independently, *before* the
        # FeatureUnion concatenates them. Normalising after the concat would
        # change the relative weight of the word and char blocks.
        return vec / norm if norm > 0 else vec


class RuntimeIntentClassifier:
    """Calibrated one-vs-rest linear SVM, evaluated in numpy."""

    def __init__(self, blob: dict):
        self.blocks = [
            TfidfBlock(
                vocabulary=b["vocabulary"],
                idf=np.asarray(b["idf"]),
                analyzer=b["analyzer"],
                min_n=b["min_n"],
                max_n=b["max_n"],
                sublinear_tf=b["sublinear_tf"],
            )
            for b in blob["blocks"]
        ]
        self.classes = np.asarray(blob["classes"])
        self.folds = [
            {
                "coef": np.asarray(f["coef"]),
                "intercept": np.asarray(f["intercept"]),
                "a": np.asarray(f["a"]),
                "b": np.asarray(f["b"]),
            }
            for f in blob["folds"]
        ]

    def features(self, text: str) -> np.ndarray:
        # Both sklearn paths normalise *before* the vectorizer (see
        # IntentClassifier.predict_proba); the analyzer's own lowercasing is not
        # a substitute, since normalize() also expands contractions and strips
        # punctuation. Missing this silently changes tokenisation.
        text = _normalize(text)
        return np.concatenate([b.transform(text) for b in self.blocks])

    def predict_proba(self, text: str) -> np.ndarray:
        x = self.features(text)
        total = np.zeros(len(self.classes), dtype=np.float64)
        for f in self.folds:
            decision = f["coef"] @ x + f["intercept"]
            # Platt scaling, expit(-(a*d + b)), written to avoid overflow.
            proba = _expit(-(f["a"] * decision + f["b"]))
            s = proba.sum()
            if s > 0:
                proba = proba / s          # OvR normalisation, per fold
            total += proba
        return total / len(self.folds)     # then averaged across folds

    def predict_one(self, text: str, k: int = 3):
        proba = self.predict_proba(text)
        order = np.argsort(proba)[::-1]
        top = [(str(self.classes[i]), float(proba[i])) for i in order[:k]]
        margin = float(proba[order[0]] - proba[order[1]]) if len(order) > 1 else 1.0
        return top[0][0], top[0][1], margin, top


def _expit(z: np.ndarray) -> np.ndarray:
    out = np.empty_like(z, dtype=np.float64)
    pos, neg = z >= 0, z < 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[neg])
    out[neg] = ez / (1.0 + ez)
    return out


class RuntimeRetriever:
    """TF-IDF -> SVD -> L2 projection plus the precomputed FAQ matrix."""

    def __init__(self, blob: dict):
        b = blob["tfidf"]
        self.tfidf = TfidfBlock(
            vocabulary=b["vocabulary"], idf=np.asarray(b["idf"]), analyzer=b["analyzer"],
            min_n=b["min_n"], max_n=b["max_n"], sublinear_tf=b["sublinear_tf"],
        )
        self.components = np.asarray(blob["components"])   # (k, n_features)
        self.matrix = np.asarray(blob["matrix"])           # (n_faqs, k)
        self.meta = blob["meta"]
        self.vocab_tokens = set(blob["vocab_tokens"])
        self.threshold = blob["threshold"]
        self.retry_threshold = blob.get("retry_threshold", blob["threshold"])
        self.coverage_threshold = blob["coverage_threshold"]
        self.intent_floor = blob["intent_floor"]
        self.stop_words = set(blob["stop_words"])

    def encode(self, text: str) -> np.ndarray:
        v = self.components @ self.tfidf.transform(_normalize(text))
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def lexical_coverage(self, text: str) -> float:
        toks = [t for t in _normalize(text).split()
                if t not in self.stop_words and len(t) > 2]
        if not toks:
            return 0.0
        return sum(t in self.vocab_tokens for t in toks) / len(toks)

    def search(self, text: str, k: int = 5, intent_filter: str | None = None):
        scores = self.matrix @ self.encode(text)
        if intent_filter:
            mask = np.array([m["intent"] == intent_filter for m in self.meta])
            if mask.any():
                scores = np.where(mask, scores, -np.inf)
        order = np.argsort(scores)[::-1][:k]
        return [
            {**self.meta[i], "score": float(scores[i]), "rank": r}
            for r, i in enumerate(order, 1)
            if np.isfinite(scores[i])
        ]

    def fallback_reason(self, hits, text, intent_confidence=None) -> str | None:
        if not hits:
            return "no_candidates"
        if self.lexical_coverage(text) < self.coverage_threshold:
            return "out_of_vocabulary"
        if hits[0]["score"] < self.threshold:
            return "low_similarity"
        if (intent_confidence is not None and intent_confidence < self.intent_floor
                and hits[0]["score"] < self.threshold + 0.15):
            return "ambiguous_intent"
        return None


_PUNCT = re.compile(r"[^a-z0-9\s]")
_WS = re.compile(r"\s+")
_CONTRACTIONS = {"can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
                 "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am"}


def _normalize(text: str) -> str:
    """Must match src.preprocess.normalize exactly."""
    import unicodedata

    text = re.sub(r"\{\{\s*(.*?)\s*\}\}", lambda m: m.group(1), str(text)).lower()
    for k, v in _CONTRACTIONS.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return _WS.sub(" ", _PUNCT.sub(" ", text)).strip()


def load_runtime(path: str) -> tuple[RuntimeIntentClassifier, RuntimeRetriever]:
    """Load the exported .npz bundle (classifier, retriever)."""
    data = np.load(path, allow_pickle=False)
    clf_blob = json.loads(str(data["classifier_json"]))
    ret_blob = json.loads(str(data["retriever_json"]))
    for i, blk in enumerate(clf_blob["blocks"]):
        blk["idf"] = data[f"clf_idf_{i}"]
    for i, f in enumerate(clf_blob["folds"]):
        f["coef"] = data[f"clf_coef_{i}"]
        f["intercept"] = data[f"clf_intercept_{i}"]
        f["a"] = data[f"clf_a_{i}"]
        f["b"] = data[f"clf_b_{i}"]
    ret_blob["tfidf"]["idf"] = data["ret_idf"]
    ret_blob["components"] = data["ret_components"]
    ret_blob["matrix"] = data["ret_matrix"]
    return RuntimeIntentClassifier(clf_blob), RuntimeRetriever(ret_blob)
