"""End-to-end inference pipeline shared by the CLI and the Lambda handler."""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field

from .entities import EntityExtractor
from .intent_model import IntentClassifier
from .rag import RagAnswerer, strip_citations
from .retriever import FaqRetriever, RetrievalHit

FALLBACK_MESSAGE = (
    "I'm not confident I have the right answer for that one. "
    "I can connect you with a human agent, or you can rephrase the question."
)


@dataclass
class BotAnswer:
    query: str
    intent: str | None
    intent_confidence: float
    answer: str
    faq_id: str | None
    retrieval_score: float
    is_fallback: bool
    fallback_reason: str | None = None
    lexical_coverage: float = 0.0
    answer_mode: str = "extractive"      # extractive | generative
    citations: list = field(default_factory=list)
    groundedness: float = 0.0
    rag_abstained_because: str | None = None
    entities: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0
    request_id: str = ""

    def to_dict(self) -> dict:
        return self.__dict__.copy()


class FaqBot:
    """Two-stage answering: classify the intent, then retrieve within it.

    Restricting retrieval to the predicted intent is what lifts top-1 accuracy
    on paraphrase-heavy corpora like Bitext, where many FAQ entries across
    intents share near-identical vocabulary. When the classifier is unsure
    (low confidence or a thin top-2 margin) we drop the filter and search the
    whole index instead of trusting a shaky label.
    """

    def __init__(
        self,
        classifier: IntentClassifier,
        retriever: FaqRetriever,
        extractor: EntityExtractor | None = None,
        rag: RagAnswerer | None = None,
        intent_confidence_floor: float = 0.35,
        intent_margin_floor: float = 0.10,
        top_k: int = 5,
    ):
        self.classifier = classifier
        self.retriever = retriever
        self.extractor = extractor or EntityExtractor()
        self.rag = rag
        self.intent_confidence_floor = intent_confidence_floor
        self.intent_margin_floor = intent_margin_floor
        self.top_k = top_k

    def answer(self, query: str, request_id: str | None = None) -> BotAnswer:
        started = time.perf_counter()
        request_id = request_id or str(uuid.uuid4())

        pred = self.classifier.predict_one(query, k=3)
        trust_intent = (
            pred.confidence >= self.intent_confidence_floor
            and pred.margin >= self.intent_margin_floor
        )
        hits: list[RetrievalHit] = self.retriever.search(
            query, k=self.top_k, intent_filter=pred.intent if trust_intent else None
        )
        # If the filtered search came back weak, retry unfiltered before giving up.
        if trust_intent and (not hits or hits[0].score < self.retriever.threshold):
            unfiltered = self.retriever.search(query, k=self.top_k)
            if unfiltered and (not hits or unfiltered[0].score > hits[0].score):
                hits = unfiltered

        reason = self.retriever.fallback_reason(hits, query, pred.confidence)
        fallback = reason is not None
        top = hits[0] if hits else None
        entities = self.extractor.extract(query)

        # Generation runs only on retrieval we already trust. If it abstains
        # for any reason we serve the canned FAQ response instead of degrading
        # to "I don't know" — a correct verbatim answer beats a refusal.
        answer_text = FALLBACK_MESSAGE if (fallback or top is None) else top.response
        mode, citations, ground, rag_reason = "extractive", [], 0.0, None
        if self.rag is not None and not fallback and top is not None:
            rag_result = self.rag.answer(query, hits)
            ground, rag_reason = rag_result.groundedness, rag_result.reason
            if not rag_result.abstained:
                answer_text = strip_citations(rag_result.text)
                citations = rag_result.citations
                mode = "generative"

        return BotAnswer(
            query=query,
            intent=pred.intent if trust_intent else None,
            intent_confidence=round(pred.confidence, 4),
            answer=answer_text,
            faq_id=None if fallback or top is None else top.faq_id,
            retrieval_score=round(top.score, 4) if top else 0.0,
            is_fallback=bool(fallback or top is None),
            fallback_reason=reason if (fallback or top is None) else None,
            lexical_coverage=round(self.retriever.lexical_coverage(query), 4),
            answer_mode=mode,
            citations=citations,
            groundedness=round(ground, 4),
            rag_abstained_because=rag_reason,
            entities=entities.as_dict(),
            candidates=[
                {"faq_id": h.faq_id, "intent": h.intent, "score": round(h.score, 4)}
                for h in hits[:3]
            ],
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            request_id=request_id,
        )
