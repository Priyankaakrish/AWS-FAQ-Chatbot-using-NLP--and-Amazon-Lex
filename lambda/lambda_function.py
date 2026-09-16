"""Lex V2 fulfillment code hook.

Deployment shape
----------------
The model artifacts are baked into a Lambda **layer** mounted at /opt/artifacts
rather than fetched from S3 per invocation. They are a few MB, they change only
on retrain, and a layer keeps warm-start latency in single-digit milliseconds.
S3 loading is kept behind ARTIFACT_BUCKET for the case where artifacts outgrow
the 250 MB layer limit (e.g. a sentence-transformers backend).

Everything expensive is module-level so it runs once per cold start, not once
per turn.

Environment variables
    ARTIFACT_DIR       default /opt/artifacts
    ARTIFACT_BUCKET    optional; S3 bucket to pull artifacts from instead
    ARTIFACT_PREFIX    optional; S3 key prefix
    INTERACTIONS_TABLE DynamoDB table for the analytics log
    METRIC_NAMESPACE   CloudWatch namespace, default FaqChatbot
    LOG_UNKNOWN_ONLY   "true" to only persist fallbacks (cost control)
"""
from __future__ import annotations

import decimal
import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

logger = logging.getLogger()
logger.setLevel(logging.INFO)

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/opt/artifacts")
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET")
ARTIFACT_PREFIX = os.environ.get("ARTIFACT_PREFIX", "artifacts/")
TABLE_NAME = os.environ.get("INTERACTIONS_TABLE")
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "FaqChatbot")
LOG_UNKNOWN_ONLY = os.environ.get("LOG_UNKNOWN_ONLY", "false").lower() == "true"

_BOT = None
_TABLE = None


# ------------------------------------------------------------------ loading --
def _download_artifacts() -> str:
    """Pull artifacts from S3 into /tmp when a layer is not used."""
    import boto3

    target = "/tmp/artifacts"
    os.makedirs(target, exist_ok=True)
    s3 = boto3.client("s3")
    for name in ("intent_model.joblib", "faq_index.joblib", "manifest.json"):
        dest = os.path.join(target, name)
        if not os.path.exists(dest):
            s3.download_file(ARTIFACT_BUCKET, f"{ARTIFACT_PREFIX}{name}", dest)
    return target


def get_bot():
    global _BOT
    if _BOT is None:
        from src.entities import EntityExtractor
        from src.intent_model import IntentClassifier
        from src.pipeline import FaqBot
        from src.retriever import FaqRetriever

        path = _download_artifacts() if ARTIFACT_BUCKET else ARTIFACT_DIR
        _BOT = FaqBot(
            IntentClassifier.load(os.path.join(path, "intent_model.joblib")),
            FaqRetriever.load(os.path.join(path, "faq_index.joblib")),
            EntityExtractor(),
        )
        logger.info("Loaded FAQ bot artifacts from %s", path)
    return _BOT


def get_table():
    global _TABLE
    if _TABLE is None and TABLE_NAME:
        import boto3

        _TABLE = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _TABLE


# ---------------------------------------------------------------- telemetry --
def emit_metrics(result, lex_intent: str) -> None:
    """CloudWatch Embedded Metric Format.

    EMF turns a structured log line into custom metrics with no extra API call
    and no added latency — important when the whole turn budget is ~200 ms.
    """
    payload = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": NAMESPACE,
                    "Dimensions": [["LexIntent"], ["PredictedIntent"]],
                    "Metrics": [
                        {"Name": "ResponseTimeMs", "Unit": "Milliseconds"},
                        {"Name": "RetrievalScore", "Unit": "None"},
                        {"Name": "IntentConfidence", "Unit": "None"},
                        {"Name": "FallbackCount", "Unit": "Count"},
                        {"Name": "AnsweredCount", "Unit": "Count"},
                    ],
                }
            ],
        },
        "LexIntent": lex_intent,
        "PredictedIntent": result.intent or "UNKNOWN",
        "ResponseTimeMs": result.latency_ms,
        "RetrievalScore": result.retrieval_score,
        "IntentConfidence": result.intent_confidence,
        "FallbackCount": 1 if result.is_fallback else 0,
        "AnsweredCount": 0 if result.is_fallback else 1,
        "FallbackReason": result.fallback_reason or "none",
    }
    print(json.dumps(payload))


def log_interaction(result, session_id: str) -> None:
    table = get_table()
    if table is None:
        return
    if LOG_UNKNOWN_ONLY and not result.is_fallback:
        return
    now = datetime.now(timezone.utc)
    try:
        table.put_item(
            Item=json.loads(
                json.dumps(
                    {
                        "pk": f"SESSION#{session_id}",
                        "sk": f"TS#{now.isoformat()}#{result.request_id}",
                        "gsi1pk": f"INTENT#{result.intent or 'UNKNOWN'}",
                        "gsi1sk": now.isoformat(),
                        "date": now.date().isoformat(),
                        "query": result.query,
                        "intent": result.intent,
                        "intent_confidence": result.intent_confidence,
                        "faq_id": result.faq_id,
                        "retrieval_score": result.retrieval_score,
                        "lexical_coverage": result.lexical_coverage,
                        "is_fallback": result.is_fallback,
                        "fallback_reason": result.fallback_reason,
                        "entities": result.entities,
                        "latency_ms": result.latency_ms,
                        # 90-day TTL keeps the analytics table from growing forever.
                        "ttl": int(now.timestamp()) + 90 * 86400,
                    }
                ),
                parse_float=decimal.Decimal,
            )
        )
    except Exception:  # telemetry must never break the conversation
        logger.exception("Failed to log interaction")


# ------------------------------------------------------------- lex plumbing --
def close(intent_name: str, message: str, session_attrs: dict,
          fulfilled: bool = True) -> dict:
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": intent_name,
                       "state": "Fulfilled" if fulfilled else "Failed"},
            "sessionAttributes": session_attrs,
        },
        "messages": [{"contentType": "PlainText", "content": message}],
    }


def elicit_slot(intent_name: str, slot: str, message: str, slots: dict,
                session_attrs: dict) -> dict:
    return {
        "sessionState": {
            "dialogAction": {"type": "ElicitSlot", "slotToElicit": slot},
            "intent": {"name": intent_name, "slots": slots, "state": "InProgress"},
            "sessionAttributes": session_attrs,
        },
        "messages": [{"contentType": "PlainText", "content": message}],
    }


def lex_slot_values(event: dict) -> dict:
    slots = (event.get("sessionState", {}).get("intent", {}).get("slots") or {})
    out = {}
    for name, payload in slots.items():
        if payload and payload.get("value", {}).get("interpretedValue"):
            out[name] = payload["value"]["interpretedValue"]
    return out


# Intents where Lambda should ask Lex to collect a slot before answering.
REQUIRED_SLOTS = {
    "TrackOrder": ("OrderNumber", "Sure — what is your order number?"),
    "CancelOrder": ("OrderNumber", "I can help with that. Which order number?"),
    "CheckInvoice": ("InvoiceNumber", "What is the invoice number?"),
}


def lambda_handler(event, context):
    logger.info("event=%s", json.dumps(event)[:2000])
    session_state = event.get("sessionState", {})
    lex_intent = session_state.get("intent", {}).get("name", "FallbackIntent")
    session_attrs = session_state.get("sessionAttributes") or {}
    utterance = event.get("inputTranscript", "") or ""
    session_id = event.get("sessionId", str(uuid.uuid4()))

    if not utterance.strip():
        return close(lex_intent, "Could you type your question again?", session_attrs, False)

    # Slot elicitation happens before retrieval: an order-status answer is
    # useless without the order number, so ask for it first.
    if lex_intent in REQUIRED_SLOTS:
        slot_name, prompt = REQUIRED_SLOTS[lex_intent]
        if slot_name not in lex_slot_values(event):
            return elicit_slot(
                lex_intent, slot_name, prompt,
                session_state.get("intent", {}).get("slots", {}), session_attrs,
            )

    try:
        result = get_bot().answer(utterance)
    except Exception:
        logger.exception("Inference failed")
        return close(lex_intent,
                     "Something went wrong on my side. Please try again in a moment.",
                     session_attrs, False)

    emit_metrics(result, lex_intent)
    log_interaction(result, session_id)

    session_attrs.update(
        {
            "lastIntent": result.intent or "UNKNOWN",
            "lastFaqId": result.faq_id or "",
            "lastRequestId": result.request_id,
            "consecutiveFallbacks": str(
                int(session_attrs.get("consecutiveFallbacks", "0")) + 1
                if result.is_fallback else 0
            ),
        }
    )

    message = result.answer
    # Two misses in a row is the point to stop guessing and offer a human.
    if int(session_attrs["consecutiveFallbacks"]) >= 2:
        message += " Would you like me to open a support ticket for you?"

    return close(lex_intent, message, session_attrs, not result.is_fallback)
