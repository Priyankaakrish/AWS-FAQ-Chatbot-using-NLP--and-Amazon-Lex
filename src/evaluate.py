"""Evaluation harness.

Produces everything the project brief asks for:

  1. Intent classification — accuracy, macro/weighted precision, recall, F1,
     per-class report, confusion matrix (CSV + PNG).
  2. Retrieval — top-1 / top-3 accuracy and MRR, scored by whether the returned
     FAQ entry carries the gold intent (Bitext has no gold FAQ id, so intent
     agreement is the retrieval label).
  3. Fallback calibration — sweep the similarity threshold and report the
     coverage/accuracy trade-off, which is how you pick `--threshold` rather
     than guessing it.
  4. Latency — mean and p95 end-to-end, the number CloudWatch will track.

    python -m src.evaluate --artifacts artifacts --out reports
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .entities import EntityExtractor
from .intent_model import IntentClassifier
from .pipeline import FaqBot
from .retriever import FaqRetriever


def _force_utf8_stdout() -> None:
    """Windows consoles default to a legacy codepage (cp1252).

    The Bitext corpus contains curly quotes, accented names and currency
    symbols, so printing a retrieved answer raises UnicodeEncodeError on a
    stock PowerShell session. Reconfiguring stdout is cheaper than asking
    every user to run `chcp 65001`.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass  # non-tty or Python < 3.7; nothing to do


def intent_metrics(y_true, y_pred, labels) -> dict:
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(p_macro),
        "recall_macro": float(r_macro),
        "f1_macro": float(f_macro),
        "precision_weighted": float(p_w),
        "recall_weighted": float(r_w),
        "f1_weighted": float(f_w),
        "n_labels": len(labels),
    }


def save_confusion_matrix(y_true, y_pred, labels, out_dir: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(
        os.path.join(out_dir, "confusion_matrix.csv")
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        norm = cm / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
        size = max(6, len(labels) * 0.42)
        fig, ax = plt.subplots(figsize=(size, size))
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=7)
        ax.set_yticks(range(len(labels)), labels, fontsize=7)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title("Intent confusion matrix (row-normalised)")
        fig.colorbar(im, fraction=0.046)
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=150)
        plt.close(fig)
    except ImportError:
        pass


def retrieval_metrics(bot: FaqBot, texts, gold_intents, k: int = 3) -> tuple[dict, list[dict]]:
    rows, rr, top1, topk, latencies = [], [], 0, 0, []
    for text, gold in zip(texts, gold_intents):
        res = bot.answer(text)
        latencies.append(res.latency_ms)
        ranks = [c["intent"] for c in res.candidates[:k]]
        hit_rank = ranks.index(gold) + 1 if gold in ranks else 0
        top1 += int(bool(ranks) and ranks[0] == gold)
        topk += int(hit_rank > 0)
        rr.append(1.0 / hit_rank if hit_rank else 0.0)
        rows.append(
            {
                "query": text,
                "gold_intent": gold,
                "pred_intent": res.intent,
                "intent_confidence": res.intent_confidence,
                "retrieved_intent": ranks[0] if ranks else None,
                "retrieval_score": res.retrieval_score,
                "lexical_coverage": res.lexical_coverage,
                "is_fallback": res.is_fallback,
                "fallback_reason": res.fallback_reason,
                "latency_ms": res.latency_ms,
            }
        )
    n = max(1, len(texts))
    metrics = {
        f"retrieval_top1_accuracy": top1 / n,
        f"retrieval_top{k}_accuracy": topk / n,
        "retrieval_mrr": float(np.mean(rr)),
        "fallback_rate": float(np.mean([r["is_fallback"] for r in rows])),
        "latency_ms_mean": float(np.mean(latencies)),
        "latency_ms_p95": float(np.percentile(latencies, 95)),
    }
    return metrics, rows


def fallback_metrics(bot: FaqBot, in_domain: list[str], ood: list[str]) -> dict:
    """How well does the bot know what it does not know?

    Treating "should fall back" as the positive class: recall is the share of
    out-of-domain questions correctly refused, and the false-alarm rate is the
    share of answerable questions we needlessly refused. Both matter — a bot
    that falls back on everything scores perfect recall and is useless.
    """
    ood_flags = [bot.answer(q).is_fallback for q in ood]
    id_flags = [bot.answer(q).is_fallback for q in in_domain]
    tp, fn = sum(ood_flags), len(ood_flags) - sum(ood_flags)
    fp, tn = sum(id_flags), len(id_flags) - sum(id_flags)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "fallback_precision": precision,
        "fallback_recall": recall,
        "fallback_f1": 2 * precision * recall / max(1e-9, precision + recall),
        "false_alarm_rate": fp / max(1, fp + tn),
        "n_ood": len(ood),
        "n_in_domain": len(in_domain),
    }


def threshold_sweep(rows: list[dict], grid=np.arange(0.10, 0.95, 0.05)) -> pd.DataFrame:
    """Coverage vs. accuracy-on-answered as the fallback threshold moves."""
    out = []
    for t in grid:
        answered = [r for r in rows if r["retrieval_score"] >= t]
        correct = [r for r in answered if r["retrieved_intent"] == r["gold_intent"]]
        out.append(
            {
                "threshold": round(float(t), 2),
                "coverage": len(answered) / max(1, len(rows)),
                "accuracy_on_answered": len(correct) / max(1, len(answered)),
                "n_answered": len(answered),
            }
        )
    return pd.DataFrame(out)


def main() -> None:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="reports")
    ap.add_argument("--sample", type=int, default=2000,
                    help="Cap on test rows used for the retrieval loop (it is O(n) queries).")
    ap.add_argument("--ood", default=None,
                    help="Optional text file, one out-of-domain question per line, "
                         "used to score fallback precision/recall.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    test = pd.read_csv(os.path.join(args.artifacts, "test.csv"))
    clf = IntentClassifier.load(os.path.join(args.artifacts, "intent_model.joblib"))
    retriever = FaqRetriever.load(os.path.join(args.artifacts, "faq_index.joblib"))
    bot = FaqBot(clf, retriever, EntityExtractor())

    # --- 1. intent classification -------------------------------------------
    y_true = test["intent"].tolist()
    y_pred = clf.predict(test["instruction"].tolist())
    labels = sorted(set(y_true) | set(y_pred))
    cls = intent_metrics(y_true, y_pred, labels)
    report = classification_report(y_true, y_pred, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(os.path.join(args.out, "intent_classification_report.csv"))
    save_confusion_matrix(y_true, y_pred, labels, args.out)

    # --- 2. retrieval + 3. thresholds + 4. latency ---------------------------
    sample = test.sample(min(args.sample, len(test)), random_state=42)
    ret, rows = retrieval_metrics(bot, sample["instruction"].tolist(), sample["intent"].tolist())
    pd.DataFrame(rows).to_csv(os.path.join(args.out, "predictions.csv"), index=False)
    sweep = threshold_sweep(rows)
    sweep.to_csv(os.path.join(args.out, "threshold_sweep.csv"), index=False)

    # Errors worth reading: confidently wrong, or answered-but-wrong.
    errors = [r for r in rows if r["retrieved_intent"] != r["gold_intent"] and not r["is_fallback"]]
    pd.DataFrame(errors).to_csv(os.path.join(args.out, "error_analysis.csv"), index=False)

    metrics = {"intent_classification": cls, "retrieval": ret, "n_test": int(len(test))}

    if args.ood and os.path.exists(args.ood):
        with open(args.ood) as fh:
            ood_qs = [ln.strip() for ln in fh if ln.strip()]
        id_qs = sample["instruction"].head(len(ood_qs) * 3).tolist()
        metrics["fallback"] = fallback_metrics(bot, id_qs, ood_qs)
    with open(os.path.join(args.out, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"\nThreshold sweep (pick the knee):\n{sweep.to_string(index=False)}")
    print(f"\nWrote reports to {args.out}/")


if __name__ == "__main__":
    main()
