"""Retrieval-augmented generation over the FAQ index.

Why this exists
---------------
The extractive path returns one canned FAQ response verbatim. That is correct
and fast, but it fails in three situations the Bitext corpus produces constantly:

  1. The user asks a compound question ("cancel my order and refund me") whose
     answer lives in two FAQ entries.
  2. The top FAQ is *related* but phrased for a different situation, so the
     verbatim response reads as a non-answer.
  3. The question needs the user's own entities woven in ("where is order
     4562781") and the canned text is generic.

RAG fixes these by synthesising from the top-k retrieved entries. The cost is
hallucination risk, which is why this module is built around three controls
rather than just a prompt:

  * **Citations.** The model must tag each claim with a FAQ id. Ids it invents
    are detectable — we know the valid set — so citation validity is a cheap
    hallucination alarm that needs no second model.
  * **Groundedness.** Token-level overlap between the answer and the retrieved
    context. Low overlap means the model wrote from its own priors.
  * **Abstention.** Generation is refused outright when retrieval is weak. RAG
    on bad context does not rescue the answer, it only makes a wrong one
    fluent, which is strictly worse for a support bot.

Generation never runs on out-of-domain queries. The fallback gates in
`retriever.py` fire first, by design: an LLM handed irrelevant context is
exactly how "confidently wrong" happens.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Protocol

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from .preprocess import normalize
from .retriever import RetrievalHit

STOP_WORDS = set(ENGLISH_STOP_WORDS)
CITATION_RE = re.compile(r"\[(FAQ-\d+)\]")

SYSTEM_PROMPT = """You are a customer-support assistant. Answer ONLY from the \
numbered FAQ entries provided. Follow these rules exactly:

1. Every factual claim must be followed by the FAQ id it came from, in square \
brackets, like [FAQ-00007].
2. If the entries do not contain the answer, reply with exactly: \
INSUFFICIENT_CONTEXT
3. Never invent policies, timeframes, amounts, URLs, or FAQ ids.
4. If the user supplied an order number or similar detail, refer to it \
naturally, but do not claim to have looked it up.
5. Be concise: two or three sentences, plain text, no markdown."""

ABSTAIN_TOKEN = "INSUFFICIENT_CONTEXT"


# ------------------------------------------------------------- generators ---
class Generator(Protocol):
    name: str

    def generate(self, system: str, prompt: str) -> str: ...


class ExtractiveGenerator:
    """Offline stand-in that performs no generation at all.

    Selects the sentences from the retrieved context that best overlap the
    query and cites their source. It exists so the whole RAG pipeline —
    prompting, citation parsing, groundedness scoring, evaluation — can be
    developed and tested with no cloud dependency, and so the metrics have a
    non-generative baseline to beat. It is trivially grounded by construction,
    which makes it a useful control: if a real LLM scores *below* this on
    groundedness, the prompt is at fault.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    def generate(self, system: str, prompt: str) -> str:
        query, blocks = _parse_prompt(prompt)
        q_tokens = _content_tokens(query)
        scored = []
        for faq_id, text in blocks:
            for sentence in re.split(r"(?<=[.!?])\s+", text):
                if len(sentence.split()) < 4:
                    continue
                overlap = len(q_tokens & _content_tokens(sentence))
                scored.append((overlap, faq_id, sentence.strip()))
        scored.sort(key=lambda t: -t[0])
        picked = [s for s in scored[: self.max_sentences] if s[0] > 0]
        if not picked:
            return ABSTAIN_TOKEN
        return " ".join(f"{sentence} [{faq_id}]" for _, faq_id, sentence in picked)


class BedrockGenerator:
    """Amazon Bedrock via the Converse API.

    Converse gives one request shape across model families, so swapping Claude
    for another provider is a config change rather than a rewrite. Temperature
    is pinned to 0: for support answers, reproducibility matters more than
    phrasing variety, and it makes the eval numbers stable run to run.
    """

    name = "bedrock"

    def __init__(
        self,
        model_id: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        region: str | None = None,
        max_tokens: int = 400,
        temperature: float = 0.0,
    ):
        import boto3  # lazy: keeps local runs free of AWS imports

        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Region is explicit and recorded. Falling back to AWS_REGION alone is
        # a trap: `aws configure` writes ~/.aws/config, not the environment, so
        # the CLI and the SDK can silently target different regions.
        self.region = (
            region
            or os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    def generate(self, system: str, prompt: str) -> str:
        resp = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={
                "maxTokens": self.max_tokens,
                "temperature": self.temperature,
            },
        )
        return "".join(
            block.get("text", "") for block in resp["output"]["message"]["content"]
        ).strip()


def get_generator(name: str, **kwargs) -> Generator:
    return {"extractive": ExtractiveGenerator, "bedrock": BedrockGenerator}[name](**kwargs)


# ------------------------------------------------------------------ helpers --
def _content_tokens(text: str) -> set[str]:
    return {t for t in normalize(text).split() if t not in STOP_WORDS and len(t) > 2}


def build_prompt(query: str, hits: list[RetrievalHit]) -> str:
    blocks = "\n\n".join(
        f"[{h.faq_id}] (intent: {h.intent}, similarity: {h.score:.2f})\n{h.response}"
        for h in hits
    )
    return f"FAQ ENTRIES:\n{blocks}\n\nUSER QUESTION:\n{query}\n\nANSWER:"


def _parse_prompt(prompt: str) -> tuple[str, list[tuple[str, str]]]:
    """Recover (query, [(faq_id, text)]) from a built prompt."""
    query = ""
    if "USER QUESTION:" in prompt:
        query = prompt.split("USER QUESTION:")[1].split("ANSWER:")[0].strip()
    blocks = []
    for chunk in prompt.split("FAQ ENTRIES:")[-1].split("USER QUESTION:")[0].split("\n\n"):
        m = re.match(r"\[(FAQ-\d+)\][^\n]*\n(.+)", chunk.strip(), re.S)
        if m:
            blocks.append((m.group(1), m.group(2).strip()))
    return query, blocks


def groundedness(answer: str, hits: list[RetrievalHit]) -> float:
    """Fraction of the answer's content tokens that appear in the context.

    Deliberately lexical. A model-based faithfulness judge is more accurate but
    needs its own eval, costs a second inference per turn, and cannot run
    offline — this catches the failure that actually matters (the model
    answering from memory instead of the retrieved FAQs) at zero cost.
    """
    answer_tokens = _content_tokens(CITATION_RE.sub("", answer))
    if not answer_tokens:
        return 0.0
    context_tokens = set()
    for h in hits:
        context_tokens |= _content_tokens(h.response) | _content_tokens(h.intent)
    return len(answer_tokens & context_tokens) / len(answer_tokens)


# -------------------------------------------------------------- the answerer --
@dataclass
class RagResult:
    text: str
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    groundedness: float = 0.0
    abstained: bool = False
    reason: str | None = None
    generator: str = ""


class RagAnswerer:
    """Wraps a generator with the guardrails that make it safe to ship."""

    def __init__(
        self,
        generator: Generator,
        min_score: float = 0.45,
        min_groundedness: float = 0.35,
        top_k: int = 4,
        require_citations: bool = True,
    ):
        self.generator = generator
        self.min_score = min_score
        self.min_groundedness = min_groundedness
        self.top_k = top_k
        self.require_citations = require_citations

    def answer(self, query: str, hits: list[RetrievalHit]) -> RagResult:
        name = getattr(self.generator, "name", "unknown")
        hits = hits[: self.top_k]

        if not hits or hits[0].score < self.min_score:
            return RagResult("", abstained=True, reason="weak_retrieval", generator=name)

        try:
            raw = self.generator.generate(SYSTEM_PROMPT, build_prompt(query, hits))
        except Exception as exc:  # never let generation break the turn
            # Record the message, not just the class. An exception *type* like
            # ResourceNotFoundException is ambiguous across a dozen causes,
            # while the message names the actual remedy ("submit use case
            # details", "request an inference profile"). Truncated because this
            # string lands in DynamoDB and CloudWatch dimensions.
            detail = str(exc).strip().replace("\n", " ")[:180]
            return RagResult("", abstained=True,
                             reason=f"generator_error:{type(exc).__name__}: {detail}",
                             generator=name)

        if not raw or ABSTAIN_TOKEN in raw:
            return RagResult("", abstained=True, reason="model_abstained", generator=name)

        valid_ids = {h.faq_id for h in hits}
        cited = CITATION_RE.findall(raw)
        invalid = [c for c in cited if c not in valid_ids]
        score = groundedness(raw, hits)

        # A fabricated FAQ id means the model is not reading the context.
        # Discard the whole answer rather than trying to repair it.
        if invalid:
            return RagResult(raw, cited, invalid, score, True, "invalid_citation", name)
        if self.require_citations and not cited:
            return RagResult(raw, cited, invalid, score, True, "no_citation", name)
        if score < self.min_groundedness:
            return RagResult(raw, cited, invalid, score, True, "low_groundedness", name)

        return RagResult(raw, cited, invalid, score, False, None, name)


def strip_citations(text: str) -> str:
    """Citations are for logging and evaluation, not for the end user."""
    return re.sub(r"\s*\[FAQ-\d+\]", "", text).strip()
