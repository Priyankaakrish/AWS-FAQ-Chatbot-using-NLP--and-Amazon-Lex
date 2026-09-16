"""Text preprocessing for the Bitext customer-support corpus.

The Bitext CSV has columns: flags, instruction, category, intent, response.
`instruction` is the user utterance and contains slot placeholders such as
{{Order Number}} or {{Person Name}} — these are gold entity annotations and we
exploit them for both slot extraction and Lex sample-utterance generation.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

PLACEHOLDER_RE = re.compile(r"\{\{\s*(.*?)\s*\}\}")

REQUIRED_COLUMNS = ("instruction", "category", "intent", "response")

CONTRACTIONS = {
    "can't": "cannot", "won't": "will not", "n't": " not", "'re": " are",
    "'s": " is", "'d": " would", "'ll": " will", "'ve": " have", "'m": " am",
}


def load_bitext(path: str, dropna: bool = True) -> pd.DataFrame:
    """Load the Bitext CSV and validate its schema."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s) {missing}. Found: {list(df.columns)}")
    if dropna:
        df = df.dropna(subset=list(REQUIRED_COLUMNS)).reset_index(drop=True)
    df["intent"] = df["intent"].str.strip()
    df["category"] = df["category"].str.strip().str.upper()
    return df


def extract_placeholders(text: str) -> list[str]:
    """Return the gold slot names annotated in an utterance."""
    return [m.strip() for m in PLACEHOLDER_RE.findall(str(text))]


def strip_placeholders(text: str, keep_label: bool = True) -> str:
    """Turn '{{Order Number}}' into 'order number' (or drop it entirely)."""
    return PLACEHOLDER_RE.sub(lambda m: m.group(1) if keep_label else " ", str(text))


def normalize(text: str, expand: bool = True) -> str:
    """Lowercase, strip accents/punctuation, collapse whitespace."""
    text = strip_placeholders(text, keep_label=True).lower()
    if expand:
        for k, v in CONTRACTIONS.items():
            text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class FaqCorpus:
    """One canonical FAQ entry per (intent, response) pair.

    Bitext ships ~1000 paraphrases per intent that share a small set of
    responses, so we deduplicate responses into FAQ entries and keep every
    paraphrase as an alternative "question view" of that entry.
    """

    frame: pd.DataFrame  # faq_id, intent, category, response, questions (list)

    @property
    def responses(self) -> list[str]:
        return self.frame["response"].tolist()

    @property
    def intents(self) -> list[str]:
        return self.frame["intent"].tolist()

    def index_text(self) -> list[str]:
        """Embedding text for each FAQ entry: intent + a few paraphrases."""
        out = []
        for _, row in self.frame.iterrows():
            qs = " ".join(row["questions"][:12])
            out.append(f"{row['intent'].replace('_', ' ')} {row['category'].lower()} {qs}")
        return out


def build_faq_corpus(df: pd.DataFrame, max_questions: int = 25) -> FaqCorpus:
    grouped = (
        df.groupby(["intent", "category", "response"], sort=False)["instruction"]
        .apply(lambda s: [normalize(x) for x in s.head(max_questions)])
        .reset_index(name="questions")
    )
    grouped.insert(0, "faq_id", [f"FAQ-{i:05d}" for i in range(len(grouped))])
    return FaqCorpus(grouped)


def normalize_series(texts: Iterable[str]) -> list[str]:
    return [normalize(t) for t in texts]
