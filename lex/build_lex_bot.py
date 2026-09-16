"""Generate an Amazon Lex V2 bot definition from the Bitext dataset.

Rather than hand-authoring 27 intents in the console, we derive them: each
Bitext intent becomes a Lex intent, its placeholder annotations become slots,
and a diverse sample of its paraphrases becomes the sample utterances.

Utterance selection is deliberately *not* random. Lex trains a classifier on
these, so near-duplicates are wasted budget. We greedily pick utterances that
are maximally dissimilar from the ones already chosen, which covers more of the
paraphrase space in the ~15 utterances per intent that Lex actually needs.

    python lex/build_lex_bot.py --data data/bitext.csv --out lex/bot_definition.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.entities import PLACEHOLDER_TO_SLOT  # noqa: E402
from src.preprocess import PLACEHOLDER_RE, load_bitext  # noqa: E402

BUILTIN_SLOT_TYPES = {
    "order_id": "AMAZON.AlphaNumeric",
    "invoice_id": "AMAZON.AlphaNumeric",
    "tracking_id": "AMAZON.AlphaNumeric",
    "customer_id": "AMAZON.AlphaNumeric",
    "person_name": "AMAZON.FirstName",
    "city": "AMAZON.City",
    "country": "AMAZON.Country",
    "date": "AMAZON.Date",
    "amount": "AMAZON.Number",
    "url": "AMAZON.FreeFormInput",
}

SLOT_PROMPTS = {
    "order_id": "What is your order number?",
    "invoice_id": "Which invoice number is this about?",
    "customer_id": "What is your customer ID?",
    "date": "What date was that?",
    "amount": "How much was the amount?",
    "product_name": "Which product is this about?",
}


def pascal(name: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[^A-Za-z0-9]+", name) if p)


def to_lex_utterance(text: str) -> str:
    """'cancel order {{Order Number}}' -> 'cancel order {OrderNumber}'."""
    def repl(m):
        slot = PLACEHOLDER_TO_SLOT.get(m.group(1).strip().lower(),
                                       re.sub(r"\W+", "_", m.group(1).strip().lower()))
        return "{" + pascal(slot) + "}"

    return re.sub(r"\s+", " ", PLACEHOLDER_RE.sub(repl, str(text))).strip()


def diverse_sample(texts: list[str], n: int) -> list[str]:
    """Greedy max-min-distance selection over TF-IDF vectors."""
    texts = list(dict.fromkeys(texts))
    if len(texts) <= n:
        return texts
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(texts)
    X = np.asarray((X / np.clip(np.sqrt(X.multiply(X).sum(axis=1)), 1e-9, None)).todense())
    chosen = [int(np.argmax(np.asarray(X).sum(axis=1)))]  # start from most central
    sims = X @ X[chosen[0]]
    while len(chosen) < n:
        nxt = int(np.argmin(sims))
        if nxt in chosen:
            break
        chosen.append(nxt)
        sims = np.maximum(sims, X @ X[nxt])
    return [texts[i] for i in chosen]


def build(df, utterances_per_intent: int = 15) -> dict:
    intents, slot_types_used = [], set()

    for intent, group in df.groupby("intent"):
        raw = group["instruction"].astype(str).tolist()
        samples = [to_lex_utterance(t) for t in diverse_sample(raw, utterances_per_intent)]
        slots_in_intent = sorted({
            PLACEHOLDER_TO_SLOT.get(p.strip().lower(), re.sub(r"\W+", "_", p.strip().lower()))
            for text in raw for p in PLACEHOLDER_RE.findall(text)
        })
        slots = []
        for s in slots_in_intent:
            slot_type = BUILTIN_SLOT_TYPES.get(s, f"{pascal(s)}Type")
            if not slot_type.startswith("AMAZON."):
                slot_types_used.add(s)
            slots.append(
                {
                    "slotName": pascal(s),
                    "slotTypeName": slot_type,
                    "valueElicitationSetting": {
                        # Optional: the code hook can still answer generic
                        # questions when the slot is absent.
                        "slotConstraint": "Optional",
                        "promptSpecification": {
                            "maxRetries": 2,
                            "messageGroupsList": [
                                {"message": {"plainTextMessage": {
                                    "value": SLOT_PROMPTS.get(s, f"What is the {s.replace('_',' ')}?")
                                }}}
                            ],
                        },
                    },
                }
            )
        intents.append(
            {
                "intentName": pascal(intent),
                "description": f"Bitext intent '{intent}' (category {group['category'].iloc[0]})",
                "sampleUtterances": [{"utterance": u} for u in samples],
                "slots": slots,
                "slotPriorities": [
                    {"priority": i + 1, "slotName": s["slotName"]} for i, s in enumerate(slots)
                ],
                "fulfillmentCodeHook": {"enabled": True},
                "dialogCodeHook": {"enabled": False},
            }
        )

    intents.append(
        {
            "intentName": "FallbackIntent",
            "parentIntentSignature": "AMAZON.FallbackIntent",
            "description": "Routes unmatched utterances to semantic FAQ retrieval in Lambda.",
            "fulfillmentCodeHook": {"enabled": True},
        }
    )

    return {
        "botName": "AwsFaqChatbot",
        "localeId": "en_US",
        "nluIntentConfidenceThreshold": 0.40,
        "description": "FAQ chatbot generated from the Bitext customer-support corpus.",
        "intents": intents,
        "slotTypes": [
            {
                "slotTypeName": f"{pascal(s)}Type",
                "valueSelectionSetting": {"resolutionStrategy": "OriginalValue"},
                "slotTypeValues": [],
            }
            for s in sorted(slot_types_used)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="lex/bot_definition.json")
    ap.add_argument("--utterances", type=int, default=15)
    args = ap.parse_args()

    df = load_bitext(args.data)
    spec = build(df, args.utterances)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(spec, fh, indent=2)
    print(f"Wrote {len(spec['intents'])} intents and "
          f"{len(spec['slotTypes'])} custom slot types to {args.out}")


if __name__ == "__main__":
    main()
