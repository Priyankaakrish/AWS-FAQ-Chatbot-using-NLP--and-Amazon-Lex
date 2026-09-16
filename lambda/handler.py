"""Lambda handler for the FAQ chatbot.

Serves **two** event shapes from one function:

  * Amazon Lex V2 fulfillment code hook (`sessionState` in the event)
  * API Gateway HTTP API, for the web frontend (`requestContext.http`)

One function rather than two because the answering logic is identical and the
model artifacts are the expensive part of a cold start — duplicating the
function would double cold starts and let the two paths drift apart.

Inference uses `src/runtime.py` (numpy only), not scikit-learn: the sklearn
stack is ~150 MB unzipped against Lambda's 250 MB limit, while the exported
`.npz` is 1.6 MB and loads in ~12 ms.

Environment variables
    ARTIFACT_DIR       default /var/task/artifacts (baked into the zip)
    ARTIFACT_BUCKET    optional S3 bucket to pull runtime.npz from instead
    ARTIFACT_KEY       S3 key, default artifacts/runtime.npz
    INTERACTIONS_TABLE DynamoDB table for the analytics log
    METRIC_NAMESPACE   CloudWatch namespace, default FaqChatbot
    RAG_GENERATOR      "bedrock" to enable generation; unset = extractive only
    RAG_MODEL_ID       Bedrock model / inference-profile id
    ALLOWED_ORIGIN     CORS origin for the frontend, default *
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

ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "/var/task/artifacts")
ARTIFACT_BUCKET = os.environ.get("ARTIFACT_BUCKET")
ARTIFACT_KEY = os.environ.get("ARTIFACT_KEY", "artifacts/runtime.npz")
TABLE_NAME = os.environ.get("INTERACTIONS_TABLE")
NAMESPACE = os.environ.get("METRIC_NAMESPACE", "FaqChatbot")
RAG_GENERATOR = os.environ.get("RAG_GENERATOR", "none").lower()
RAG_MODEL_ID = os.environ.get("RAG_MODEL_ID")
ALLOWED_ORIGIN = os.environ.get("ALLOWED_ORIGIN", "*")

FALLBACK_MESSAGE = (
    "I'm not confident I have the right answer for that one. "
    "I can connect you with a human agent, or you can rephrase the question."
)

_CLF = _RET = _EXTRACTOR = _RAG = None
_TABLE = None


# ------------------------------------------------------------------ loading --
def _artifact_path() -> str:
    if not ARTIFACT_BUCKET:
        return os.path.join(ARTIFACT_DIR, "runtime.npz")
    import boto3

    dest = "/tmp/runtime.npz"
    if not os.path.exists(dest):
        boto3.client("s3").download_file(ARTIFACT_BUCKET, ARTIFACT_KEY, dest)
    return dest


def _load():
    """Module-level so it runs once per cold start, not once per request."""
    global _CLF, _RET, _EXTRACTOR, _RAG
    if _CLF is None:
        from src.entities import EntityExtractor
        from src.runtime import load_runtime

        started = time.perf_counter()
        _CLF, _RET = load_runtime(_artifact_path())
        _EXTRACTOR = EntityExtractor()
        logger.info("Loaded runtime in %.1f ms", (time.perf_counter() - started) * 1000)

        if RAG_GENERATOR not in ("none", ""):
            try:
                from src.rag import RagAnswerer, get_generator

                kwargs = {"model_id": RAG_MODEL_ID} if RAG_MODEL_ID else {}
                _RAG = RagAnswerer(get_generator(RAG_GENERATOR, **kwargs))
                logger.info("RAG enabled (%s)", RAG_GENERATOR)
            except Exception:
                # A broken generator must not take the whole bot down; the
                # extractive path is a complete product on its own.
                logger.exception("RAG init failed — continuing extractive-only")
    return _CLF, _RET, _EXTRACTOR, _RAG


def get_table():
    global _TABLE
    if _TABLE is None and TABLE_NAME:
        import boto3

        _TABLE = boto3.resource("dynamodb").Table(TABLE_NAME)
    return _TABLE


# ----------------------------------------------------------------- answering --
def answer(query: str) -> dict:
    clf, ret, extractor, rag = _load()
    started = time.perf_counter()

    intent, confidence, margin, top_k = clf.predict_one(query, k=3)
    trust = confidence >= 0.35 and margin >= 0.10
    hits = ret.search(query, k=5, intent_filter=intent if trust else None)
    if trust and (not hits or hits[0]["score"] < ret.retry_threshold):
        unfiltered = ret.search(query, k=5)
        if unfiltered and (not hits or unfiltered[0]["score"] > hits[0]["score"]):
            hits = unfiltered

    reason = ret.fallback_reason(hits, query, confidence)
    fallback = reason is not None
    top = hits[0] if hits else None

    text = FALLBACK_MESSAGE if (fallback or top is None) else top["response"]
    mode, citations, groundedness, rag_reason = "extractive", [], 0.0, None

    if rag is not None and not fallback and top is not None:
        from src.retriever import RetrievalHit
        from src.rag import strip_citations

        typed = [RetrievalHit(h["faq_id"], h["intent"], h["response"], h["score"], h["rank"])
                 for h in hits]
        result = rag.answer(query, typed)
        groundedness, rag_reason = result.groundedness, result.reason
        if not result.abstained:
            text, citations, mode = strip_citations(result.text), result.citations, "generative"

    return {
        "query": query,
        "answer": text,
        "intent": intent if trust else None,
        "intent_confidence": round(float(confidence), 4),
        "faq_id": None if fallback or top is None else top["faq_id"],
        "retrieval_score": round(float(top["score"]), 4) if top else 0.0,
        "lexical_coverage": round(ret.lexical_coverage(query), 4),
        "is_fallback": bool(fallback or top is None),
        "fallback_reason": reason,
        "answer_mode": mode,
        "citations": citations,
        "groundedness": round(float(groundedness), 4),
        "rag_abstained_because": rag_reason,
        "entities": extractor.extract(query).as_dict(),
        "alternatives": [{"intent": i, "confidence": round(float(c), 4)} for i, c in top_k],
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "request_id": str(uuid.uuid4()),
    }


# ---------------------------------------------------------------- telemetry --
def emit_metrics(res: dict, channel: str) -> None:
    """CloudWatch Embedded Metric Format: metrics from a log line, no API call."""
    print(json.dumps({
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [{
                "Namespace": NAMESPACE,
                "Dimensions": [["Channel"], ["PredictedIntent"]],
                "Metrics": [
                    {"Name": "ResponseTimeMs", "Unit": "Milliseconds"},
                    {"Name": "RetrievalScore", "Unit": "None"},
                    {"Name": "IntentConfidence", "Unit": "None"},
                    {"Name": "FallbackCount", "Unit": "Count"},
                    {"Name": "AnsweredCount", "Unit": "Count"},
                    {"Name": "GeneratedCount", "Unit": "Count"},
                ],
            }],
        },
        "Channel": channel,
        "PredictedIntent": res["intent"] or "UNKNOWN",
        "ResponseTimeMs": res["latency_ms"],
        "RetrievalScore": res["retrieval_score"],
        "IntentConfidence": res["intent_confidence"],
        "FallbackCount": 1 if res["is_fallback"] else 0,
        "AnsweredCount": 0 if res["is_fallback"] else 1,
        "GeneratedCount": 1 if res["answer_mode"] == "generative" else 0,
        "FallbackReason": res["fallback_reason"] or "none",
    }))


def log_interaction(res: dict, session_id: str, channel: str) -> None:
    table = get_table()
    if table is None:
        return
    now = datetime.now(timezone.utc)
    try:
        table.put_item(Item=json.loads(json.dumps({
            "pk": f"SESSION#{session_id}",
            "sk": f"TS#{now.isoformat()}#{res['request_id']}",
            "gsi1pk": f"INTENT#{res['intent'] or 'UNKNOWN'}",
            "gsi1sk": now.isoformat(),
            "date": now.date().isoformat(),
            "channel": channel,
            **{k: res[k] for k in (
                "query", "answer", "intent", "intent_confidence", "faq_id",
                "retrieval_score", "lexical_coverage", "is_fallback",
                "fallback_reason", "answer_mode", "citations", "groundedness",
                "entities", "latency_ms")},
            "ttl": int(now.timestamp()) + 90 * 86400,
        }), parse_float=decimal.Decimal))
    except Exception:  # telemetry must never break the conversation
        logger.exception("Failed to log interaction")


# ------------------------------------------------------------------- shapes --
def _cors() -> dict:
    return {
        "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
        "Access-Control-Allow-Headers": "content-type",
        "Access-Control-Allow-Methods": "POST,OPTIONS",
        "Content-Type": "application/json",
    }


def _http_response(status: int, body: dict) -> dict:
    return {"statusCode": status, "headers": _cors(), "body": json.dumps(body)}


def handle_http(event: dict) -> dict:
    method = event.get("requestContext", {}).get("http", {}).get("method", "POST")
    if method == "OPTIONS":
        return {"statusCode": 204, "headers": _cors(), "body": ""}

    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return _http_response(400, {"error": "invalid JSON body"})

    query = (body.get("message") or body.get("query") or "").strip()
    if not query:
        return _http_response(400, {"error": "missing 'message'"})
    if len(query) > 1000:
        return _http_response(400, {"error": "message too long"})

    session_id = body.get("sessionId") or str(uuid.uuid4())
    res = answer(query)
    emit_metrics(res, "web")
    log_interaction(res, session_id, "web")
    return _http_response(200, {**res, "sessionId": session_id})


def _close(intent_name: str, message: str, attrs: dict, fulfilled: bool) -> dict:
    return {
        "sessionState": {
            "dialogAction": {"type": "Close"},
            "intent": {"name": intent_name, "state": "Fulfilled" if fulfilled else "Failed"},
            "sessionAttributes": attrs,
        },
        "messages": [{"contentType": "PlainText", "content": message}],
    }


def handle_lex(event: dict) -> dict:
    state = event.get("sessionState", {})
    lex_intent = state.get("intent", {}).get("name", "FallbackIntent")
    attrs = state.get("sessionAttributes") or {}
    query = (event.get("inputTranscript") or "").strip()
    session_id = event.get("sessionId", str(uuid.uuid4()))

    if not query:
        return _close(lex_intent, "Could you type your question again?", attrs, False)

    res = answer(query)
    emit_metrics(res, "lex")
    log_interaction(res, session_id, "lex")

    consecutive = int(attrs.get("consecutiveFallbacks", "0")) + 1 if res["is_fallback"] else 0
    attrs.update({
        "lastIntent": res["intent"] or "UNKNOWN",
        "lastFaqId": res["faq_id"] or "",
        "consecutiveFallbacks": str(consecutive),
    })

    message = res["answer"]
    if consecutive >= 2:
        message += " Would you like me to open a support ticket for you?"
    return _close(lex_intent, message, attrs, not res["is_fallback"])


def lambda_handler(event, context):
    try:
        if "sessionState" in event or "inputTranscript" in event:
            return handle_lex(event)
        if "requestContext" in event or "body" in event:
            return handle_http(event)
        # Direct invoke (console test, smoke checks)
        return answer(event.get("query", ""))
    except Exception:
        logger.exception("Unhandled error")
        if "sessionState" in event:
            return _close(
                event.get("sessionState", {}).get("intent", {}).get("name", "FallbackIntent"),
                "Something went wrong on my side. Please try again.", {}, False)
        return _http_response(500, {"error": "internal error"})
