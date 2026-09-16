"""Train the intent classifier and build the FAQ retrieval index.

    python -m src.train --data data/bitext.csv --backend lsa --out artifacts/
"""
from __future__ import annotations

import argparse
import json
import os

from sklearn.model_selection import train_test_split

from .entities import slot_inventory
from .intent_model import IntentClassifier
from .preprocess import build_faq_corpus, load_bitext
from .retriever import FaqRetriever, get_backend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--backend", default="lsa", choices=["lsa", "sbert", "bedrock"])
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=0.45,
                    help="Fallback bar: below this similarity the bot refuses to answer.")
    ap.add_argument("--retry-threshold", type=float, default=0.45,
                    help="Separate bar for re-searching the whole index when the "
                         "intent-filtered search looks weak. Keep this low.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_bitext(args.data)
    print(f"Loaded {len(df):,} rows | {df['intent'].nunique()} intents | "
          f"{df['category'].nunique()} categories")

    # Stratified split held out for evaluate.py — saved so metrics are reproducible.
    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed, stratify=df["intent"]
    )
    train_df.to_csv(os.path.join(args.out, "train.csv"), index=False)
    test_df.to_csv(os.path.join(args.out, "test.csv"), index=False)

    print("Fitting intent classifier...")
    clf = IntentClassifier().fit(train_df["instruction"], train_df["intent"])
    clf.save(os.path.join(args.out, "intent_model.joblib"))

    print(f"Building FAQ index (backend={args.backend})...")
    corpus = build_faq_corpus(train_df)
    meta = [
        {"faq_id": r.faq_id, "intent": r.intent, "category": r.category, "response": r.response}
        for r in corpus.frame.itertuples()
    ]
    retriever = FaqRetriever(get_backend(args.backend), threshold=args.threshold,
                             retry_threshold=args.retry_threshold)
    retriever.fit(corpus.index_text(), meta)
    retriever.save(os.path.join(args.out, "faq_index.joblib"))

    manifest = {
        "backend": args.backend,
        "n_rows": int(len(df)),
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "n_intents": int(df["intent"].nunique()),
        "n_faq_entries": int(len(corpus.frame)),
        "intents": sorted(df["intent"].unique().tolist()),
        "categories": sorted(df["category"].unique().tolist()),
        "slot_inventory": slot_inventory(df["instruction"]),
        "threshold": args.threshold,
        "retry_threshold": args.retry_threshold,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"Saved {len(corpus.frame)} FAQ entries and artifacts to {args.out}/")


if __name__ == "__main__":
    main()
