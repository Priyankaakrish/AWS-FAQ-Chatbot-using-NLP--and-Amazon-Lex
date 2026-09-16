"""Chatbot analytics — the six things the project tracks.

Reads either the local evaluation output (reports/predictions.csv) or the live
DynamoDB interactions table, so the same report works before and after deploy.

    python scripts/analytics.py --source reports/predictions.csv
    python scripts/analytics.py --source dynamodb --table faq-chatbot-interactions --days 7
"""
from __future__ import annotations

import argparse
import json

import pandas as pd


def load_dynamodb(table_name: str, days: int) -> pd.DataFrame:
    """Scan recent interactions.

    A scan is fine for a few thousand rows/day. Past that, query the
    INTENT#<name> GSI per intent, or point Athena at a Kinesis Firehose copy —
    scanning a hot table for dashboards will eat your read capacity.
    """
    import boto3
    from boto3.dynamodb.conditions import Attr
    from datetime import date, timedelta

    cutoff = (date.today() - timedelta(days=days)).isoformat()
    table = boto3.resource("dynamodb").Table(table_name)
    items, kwargs = [], {"FilterExpression": Attr("date").gte(cutoff)}
    while True:
        resp = table.scan(**kwargs)
        items.extend(resp.get("Items", []))
        if "LastEvaluatedKey" not in resp:
            break
        kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
    return pd.DataFrame(items)


def report(df: pd.DataFrame) -> dict:
    df = df.copy()
    for col in ("latency_ms", "retrieval_score", "intent_confidence"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "is_fallback" in df:
        df["is_fallback"] = df["is_fallback"].astype(str).str.lower().isin(["true", "1"])

    intent_col = "intent" if "intent" in df else "pred_intent"
    answered = df[~df["is_fallback"]] if "is_fallback" in df else df
    failed = df[df["is_fallback"]] if "is_fallback" in df else df.iloc[0:0]

    out = {
        "total_queries": int(len(df)),
        # 1. most frequently asked questions
        "top_faqs": (
            df["faq_id"].value_counts().head(10).to_dict() if "faq_id" in df else {}
        ),
        "top_questions": df["query"].value_counts().head(10).to_dict() if "query" in df else {},
        # 2. intent distribution
        "intent_distribution": df[intent_col].fillna("UNKNOWN").value_counts().to_dict(),
        # 3. failed / unknown queries
        "fallback_rate": float(df["is_fallback"].mean()) if "is_fallback" in df else 0.0,
        "fallback_reasons": (
            failed["fallback_reason"].fillna("unknown").value_counts().to_dict()
            if "fallback_reason" in df else {}
        ),
        "sample_failed_queries": failed["query"].head(20).tolist() if "query" in failed else [],
        # 4. response accuracy (only computable when gold labels exist)
        "accuracy": (
            float((df["gold_intent"] == df["retrieved_intent"]).mean())
            if {"gold_intent", "retrieved_intent"} <= set(df.columns) else None
        ),
        # 5. response time
        "latency_ms_mean": float(df["latency_ms"].mean()) if "latency_ms" in df else None,
        "latency_ms_p50": float(df["latency_ms"].quantile(0.50)) if "latency_ms" in df else None,
        "latency_ms_p95": float(df["latency_ms"].quantile(0.95)) if "latency_ms" in df else None,
        # 6. user satisfaction (thumbs written back by the client)
        "satisfaction_rate": (
            float(pd.to_numeric(df["helpful"], errors="coerce").mean())
            if "helpful" in df else None
        ),
        "mean_confidence_answered": (
            float(answered["retrieval_score"].mean()) if "retrieval_score" in answered else None
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="reports/predictions.csv",
                    help="CSV path, or the literal string 'dynamodb'")
    ap.add_argument("--table", default=None)
    ap.add_argument("--days", type=int, default=7)
    ap.add_argument("--out", default="reports/analytics.json")
    args = ap.parse_args()

    if args.source == "dynamodb":
        if not args.table:
            raise SystemExit("--table is required when --source dynamodb")
        df = load_dynamodb(args.table, args.days)
    else:
        df = pd.read_csv(args.source)

    result = report(df)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2, default=str)
    print(json.dumps(result, indent=2, default=str)[:3000])
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
