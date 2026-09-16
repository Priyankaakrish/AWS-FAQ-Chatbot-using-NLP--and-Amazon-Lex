"""Local REPL / batch tester.

    python -m src.cli --artifacts artifacts
    python -m src.cli --artifacts artifacts --query "where is order 88231?"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

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


def load_bot(artifacts: str) -> FaqBot:
    return FaqBot(
        IntentClassifier.load(os.path.join(artifacts, "intent_model.joblib")),
        FaqRetriever.load(os.path.join(artifacts, "faq_index.joblib")),
        EntityExtractor(),
    )


def main() -> None:
    _force_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--query", action="append", help="Run once per --query and exit.")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    bot = load_bot(args.artifacts)

    def show(q: str) -> None:
        res = bot.answer(q)
        if args.json:
            print(json.dumps(res.to_dict(), indent=2))
            return
        tag = "FALLBACK" if res.is_fallback else res.intent
        print(f"\nQ: {q}")
        print(f"   [{tag}] intent_conf={res.intent_confidence:.3f} "
              f"sim={res.retrieval_score:.3f} {res.latency_ms:.1f}ms")
        if res.entities["slots"]:
            print(f"   slots: {res.entities['slots']}")
        print(f"A: {res.answer}")

    if args.query:
        for q in args.query:
            show(q)
        return

    print("Type a question (blank line to quit).")
    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        show(q)


if __name__ == "__main__":
    main()
