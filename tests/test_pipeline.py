"""Test suite.

Fixtures train a real (tiny) model once per session rather than mocking it —
the behaviours worth protecting here are the fallback gates and the two-stage
routing, and both are properties of a fitted model, not of the plumbing.

    pytest -q tests/
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entities import EntityExtractor, slot_inventory  # noqa: E402
from src.intent_model import IntentClassifier  # noqa: E402
from src.pipeline import FaqBot  # noqa: E402
from src.preprocess import (  # noqa: E402
    build_faq_corpus,
    extract_placeholders,
    load_bitext,
    normalize,
    strip_placeholders,
)
from src.retriever import FaqRetriever, LsaBackend  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "sample_bitext.csv")


@pytest.fixture(scope="session")
def df():
    if not os.path.exists(DATA):
        pytest.skip("run `make sample` first")
    return load_bitext(DATA)


@pytest.fixture(scope="session")
def bot(df):
    clf = IntentClassifier().fit(df["instruction"], df["intent"])
    corpus = build_faq_corpus(df)
    meta = [
        {"faq_id": r.faq_id, "intent": r.intent, "category": r.category, "response": r.response}
        for r in corpus.frame.itertuples()
    ]
    retriever = FaqRetriever(LsaBackend()).fit(corpus.index_text(), meta)
    return FaqBot(clf, retriever, EntityExtractor())


# ------------------------------------------------------------ preprocessing --
def test_load_validates_schema(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n")
    with pytest.raises(ValueError, match="Missing expected column"):
        load_bitext(str(bad))


def test_placeholder_round_trip():
    text = "cancel order {{Order Number}} for {{Person Name}}"
    assert extract_placeholders(text) == ["Order Number", "Person Name"]
    assert "{{" not in strip_placeholders(text)


def test_normalize_strips_punctuation_and_case():
    assert normalize("Where's my ORDER, please?!") == "where is my order please"


def test_faq_corpus_dedupes_responses(df):
    corpus = build_faq_corpus(df)
    # 1,120 paraphrase rows collapse to one FAQ entry per distinct response.
    assert len(corpus.frame) < len(df)
    assert corpus.frame["faq_id"].is_unique


def test_slot_inventory_maps_bitext_labels(df):
    inv = slot_inventory(df["instruction"])
    assert "order_id" in inv and inv["order_id"] > 0


# ----------------------------------------------------------------- entities --
@pytest.mark.parametrize(
    "text,slot,expected",
    [
        ("where is my order 4562781?", "order_id", "4562781"),
        ("order #ORD12345 is late", "order_id", "ORD12345"),
        ("check invoice INV-9931 please", "invoice_id", "9931"),
        ("email me at a.b+x@example.co.uk", "email", "a.b+x@example.co.uk"),
        ("refund me $45.99 now", "amount", "$45.99"),
        ("it shipped on 2026-03-14", "date", "2026-03-14"),
        ("delivered yesterday", "date", "yesterday"),
    ],
)
def test_entity_extraction(text, slot, expected):
    assert EntityExtractor().extract(text).slots.get(slot) == expected


def test_bare_order_number_after_order_verb():
    assert EntityExtractor().extract("my package 998877 never arrived").slots["order_id"] == "998877"


def test_product_catalog_matching():
    ex = EntityExtractor(product_catalog=["Widget Pro", "Widget"])
    out = ex.extract("my Widget Pro broke")
    assert out.slots["product_name"] == "widget pro"  # longest match wins


# ---------------------------------------------------------------- retrieval --
def test_known_question_is_answered(bot):
    res = bot.answer("where is my order 4562781?")
    assert not res.is_fallback
    assert res.intent == "track_order"
    assert res.entities["slots"]["order_id"] == "4562781"


@pytest.mark.parametrize(
    "query",
    [
        "what is the airspeed velocity of an unladen swallow",
        "how do i install a carburetor on a hydroponic tomato",
        "who won the world cup in 1998",
        "explain quantum entanglement to me",
    ],
)
def test_out_of_domain_triggers_fallback(bot, query):
    """Regression guard for the bug this project's design turns on.

    These queries score 0.75-0.86 cosine against unrelated FAQs — a similarity
    threshold alone answers them confidently. If this test starts failing, the
    lexical-coverage gate has been weakened.
    """
    res = bot.answer(query)
    assert res.is_fallback, f"{query!r} answered with sim={res.retrieval_score}"
    assert res.fallback_reason in {"out_of_vocabulary", "low_similarity", "ambiguous_intent"}


def test_fallback_returns_no_faq_id(bot):
    res = bot.answer("how many moons does jupiter have")
    assert res.faq_id is None
    assert "not confident" in res.answer.lower()


def test_lexical_coverage_bounds(bot):
    assert bot.retriever.lexical_coverage("xyzzy plugh frobnicate") == 0.0
    assert bot.retriever.lexical_coverage("cancel my order") > 0.5


def test_intent_filter_restricts_candidates(bot):
    hits = bot.retriever.search("i need help", k=5, intent_filter="get_refund")
    assert hits and all(h.intent == "get_refund" for h in hits)


def test_scores_are_cosine_bounded(bot):
    for h in bot.retriever.search("cancel my order", k=5):
        assert -1.01 <= h.score <= 1.01


def test_latency_is_recorded(bot):
    assert bot.answer("how do i reset my password").latency_ms > 0


def test_empty_query_does_not_crash(bot):
    assert bot.answer("").is_fallback


# --------------------------------------------------- persistence round-trip --
def test_artifacts_round_trip(bot, tmp_path):
    ipath, rpath = tmp_path / "m.joblib", tmp_path / "i.joblib"
    bot.classifier.save(str(ipath))
    bot.retriever.save(str(rpath))
    reloaded = FaqBot(IntentClassifier.load(str(ipath)), FaqRetriever.load(str(rpath)))
    a, b = bot.answer("where is my order 123456"), reloaded.answer("where is my order 123456")
    assert a.answer == b.answer and a.intent == b.intent


# ------------------------------------------------------------ lex plumbing --
def test_lex_utterance_conversion():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "lex"))
    from build_lex_bot import diverse_sample, to_lex_utterance

    assert to_lex_utterance("cancel order {{Order Number}}") == "cancel order {OrderId}"
    picked = diverse_sample(["cancel my order", "cancel my order", "reset my password"], 2)
    assert len(picked) == 2  # duplicates collapsed before selection


def test_lex_close_response_shape():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "lambda"))
    from lambda_function import close, elicit_slot

    r = close("TrackOrder", "hello", {"k": "v"})
    assert r["sessionState"]["dialogAction"]["type"] == "Close"
    assert r["sessionState"]["intent"]["state"] == "Fulfilled"
    assert r["messages"][0]["content"] == "hello"

    e = elicit_slot("TrackOrder", "OrderNumber", "which order?", {}, {})
    assert e["sessionState"]["dialogAction"]["slotToElicit"] == "OrderNumber"
    assert e["sessionState"]["intent"]["state"] == "InProgress"


def test_lex_slot_extraction_from_event():
    from lambda_function import lex_slot_values

    event = {"sessionState": {"intent": {"slots": {
        "OrderNumber": {"value": {"interpretedValue": "4562781"}},
        "Empty": None,
    }}}}
    assert lex_slot_values(event) == {"OrderNumber": "4562781"}
