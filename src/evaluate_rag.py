"""RAG evaluation.

Retrieval metrics do not tell you whether the *generated* answer is faithful,
so this is a separate harness with its own questions:

  1. **Groundedness** — how much of each answer is supported by the retrieved
     context. Reported as a distribution, not a mean: one hallucinated answer
     in a hundred matters more to a support bot than a two-point shift in the
     average.
  2. **Citation validity** — share of emitted FAQ ids that actually exist in
     the context. Anything above zero is a hallucinating model.
  3. **Abstention** — how often generation is refused, broken down by reason.
     Some abstention is healthy; all of it means the thresholds are too tight
     and you are paying for an LLM that never speaks.
  4. **Out-of-domain hallucination** — the number that decides whether this is
     shippable. Feed it questions with no answer in the KB and count how often
     it generates anyway. Target is zero.
  5. **Lift over extractive** — how often generation actually changed the
     answer. If it rarely does, RAG is added cost and latency for nothing.

    python -m src.evaluate_rag --artifacts artifacts --out reports
    python -m src.evaluate_rag --artifacts artifacts --generator bedrock --sample 200
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

from .cli import load_bot
from .rag import RagAnswerer, get_generator


def _force_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def evaluate_in_domain(bot, queries, gold_intents) -> tuple[dict, list[dict]]:
    rows = []
    for query, gold in zip(queries, gold_intents):
        # Extractive baseline first, then the same query with RAG enabled, so
        # "lift" compares two answers to the identical retrieved context.
        rag, bot.rag = bot.rag, None
        base = bot.answer(query)
        bot.rag = rag
        res = bot.answer(query)
        rows.append(
            {
                "query": query,
                "gold_intent": gold,
                "mode": res.answer_mode,
                "groundedness": res.groundedness,
                "n_citations": len(res.citations),
                "citations": ";".join(res.citations),
                "abstained_because": res.rag_abstained_because,
                "changed_answer": res.answer.strip() != base.answer.strip(),
                "latency_ms": res.latency_ms,
                "answer": res.answer,
            }
        )

    df = pd.DataFrame(rows)
    generated = df[df["mode"] == "generative"]
    g = generated["groundedness"]
    metrics = {
        "n": int(len(df)),
        "generation_rate": float((df["mode"] == "generative").mean()),
        "abstention_rate": float((df["mode"] != "generative").mean()),
        "abstention_reasons": dict(Counter(df["abstained_because"].dropna())),
        "groundedness_mean": float(g.mean()) if len(g) else None,
        "groundedness_p10": float(g.quantile(0.10)) if len(g) else None,
        "groundedness_min": float(g.min()) if len(g) else None,
        "pct_below_0.5_grounded": float((g < 0.5).mean()) if len(g) else None,
        "mean_citations_per_answer": float(generated["n_citations"].mean()) if len(generated) else None,
        "uncited_answers": int((generated["n_citations"] == 0).sum()) if len(generated) else 0,
        "multi_source_rate": float((generated["n_citations"] > 1).mean()) if len(generated) else None,
        "lift_over_extractive": float(df["changed_answer"].mean()),
        "latency_ms_mean": float(df["latency_ms"].mean()),
        "latency_ms_p95": float(np.percentile(df["latency_ms"], 95)),
    }
    return metrics, rows


def evaluate_out_of_domain(bot, queries) -> tuple[dict, list[dict]]:
    """The safety number. Generation on OOD input should never happen."""
    rows = []
    for query in queries:
        res = bot.answer(query)
        rows.append(
            {
                "query": query,
                "mode": res.answer_mode,
                "is_fallback": res.is_fallback,
                "fallback_reason": res.fallback_reason,
                "groundedness": res.groundedness,
                "answer": res.answer,
            }
        )
    df = pd.DataFrame(rows)
    generated = df[df["mode"] == "generative"]
    return {
        "n_ood": int(len(df)),
        "refusal_rate": float(df["is_fallback"].mean()),
        "ood_generation_count": int(len(generated)),
        "ood_hallucination_rate": float(len(generated) / max(1, len(df))),
        "leaked_queries": generated["query"].tolist(),
    }, rows


def main() -> None:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--generator", default="extractive", choices=["extractive", "bedrock"])
    ap.add_argument("--model-id", default=None, help="Bedrock model id (bedrock generator only)")
    ap.add_argument("--region", default=None, help="AWS region for Bedrock (default us-east-1)")
    ap.add_argument("--sample", type=int, default=300,
                    help="In-domain queries to evaluate. Keep small for bedrock — "
                         "each query is a billed inference call.")
    ap.add_argument("--ood", default="data/ood_questions.txt")
    ap.add_argument("--min-groundedness", type=float, default=0.35)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    gen_kwargs = {}
    if args.generator == "bedrock":
        if args.model_id:
            gen_kwargs["model_id"] = args.model_id
        if args.region:
            gen_kwargs["region"] = args.region
    bot = load_bot(args.artifacts)
    bot.rag = RagAnswerer(get_generator(args.generator, **gen_kwargs),
                          min_groundedness=args.min_groundedness)

    test = pd.read_csv(os.path.join(args.artifacts, "test.csv"))
    sample = test.sample(min(args.sample, len(test)), random_state=42)

    print(f"Evaluating RAG (generator={args.generator}) on {len(sample)} in-domain queries...")
    in_metrics, in_rows = evaluate_in_domain(
        bot, sample["instruction"].tolist(), sample["intent"].tolist()
    )

    ood_metrics, ood_rows = {}, []
    if os.path.exists(args.ood):
        with open(args.ood, encoding="utf-8") as fh:
            ood = [ln.strip() for ln in fh if ln.strip()]
        print(f"Evaluating {len(ood)} out-of-domain queries...")
        ood_metrics, ood_rows = evaluate_out_of_domain(bot, ood)

    pd.DataFrame(in_rows).to_csv(os.path.join(args.out, "rag_predictions.csv"), index=False)
    if ood_rows:
        pd.DataFrame(ood_rows).to_csv(os.path.join(args.out, "rag_ood.csv"), index=False)

    metrics = {"generator": args.generator, "in_domain": in_metrics, "out_of_domain": ood_metrics}
    with open(os.path.join(args.out, "rag_metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    if ood_metrics.get("ood_generation_count", 0) > 0:
        print("\nWARNING: the model generated answers for out-of-domain questions. "
              "Raise --min-groundedness or the retriever threshold before deploying.")
    print(f"\nWrote RAG reports to {args.out}/")


if __name__ == "__main__":
    main()
