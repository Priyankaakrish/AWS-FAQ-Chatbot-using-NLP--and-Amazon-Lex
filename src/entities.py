"""Entity / slot extraction.

Bitext annotates slots inline as {{Order Number}}, {{Person Name}} etc. Those
labels give us the slot inventory for free; this module supplies the runtime
extractors that recover the same slots from real, unannotated user text.

Lex fills slots itself for utterances that route through a configured intent,
but Lambda still needs extraction for (a) free-text fallback queries that never
hit a slot-bearing intent, and (b) validating what Lex captured.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .preprocess import extract_placeholders

# ---------------------------------------------------------------- patterns ---
PATTERNS: dict[str, re.Pattern] = {
    "order_id": re.compile(r"\b(?:order|ord)[\s#:-]*([A-Z]{0,3}\d{4,12})\b", re.I),
    "invoice_id": re.compile(r"\b(?:invoice|inv)[\s#:-]*([A-Z]{0,3}\d{3,12})\b", re.I),
    "customer_id": re.compile(r"\b(?:customer|cust|account)[\s#:-]*(?:id|number|no)?[\s#:-]*([A-Z]{0,3}\d{3,12})\b", re.I),
    "tracking_id": re.compile(r"\b(?:tracking|track)[\s#:-]*(?:number|no|id)?[\s#:-]*([A-Z0-9]{8,20})\b", re.I),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
    "amount": re.compile(r"(?:[$£€]\s?\d+(?:[.,]\d{1,2})?)|(?:\b\d+(?:[.,]\d{1,2})?\s?(?:usd|eur|gbp|dollars?|euros?)\b)", re.I),
    "date": re.compile(
        r"\b(?:\d{4}-\d{2}-\d{2}"
        r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
        r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?(?:,?\s*\d{4})?"
        r"|\d{1,2}(?:st|nd|rd|th)?\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*(?:,?\s*\d{4})?"
        r"|yesterday|today|tomorrow|last\s+(?:week|month|year)|this\s+(?:week|month))\b",
        re.I,
    ),
}

# Bitext placeholder label -> canonical slot name.
PLACEHOLDER_TO_SLOT = {
    "order number": "order_id",
    "invoice number": "invoice_id",
    "customer support phone number": "support_phone",
    "person name": "person_name",
    "account type": "account_type",
    "account category": "account_type",
    "delivery city": "city",
    "delivery country": "country",
    "refund amount": "amount",
    "money amount": "amount",
    "date": "date",
    "online order interaction": "channel",
    "online customer support channel": "channel",
    "website url": "url",
    "product name": "product_name",
    "tracking number": "tracking_id",
}


@dataclass
class ExtractedEntities:
    slots: dict[str, str] = field(default_factory=dict)
    products: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


class EntityExtractor:
    def __init__(self, product_catalog: list[str] | None = None):
        self.product_catalog = [p.lower() for p in (product_catalog or [])]

    def extract(self, text: str) -> ExtractedEntities:
        slots: dict[str, str] = {}
        for slot, pattern in PATTERNS.items():
            m = pattern.search(text)
            if m:
                slots[slot] = (m.group(1) if m.groups() else m.group(0)).strip()

        # Bare identifier fallback: "where is 4562781?" after an order-ish verb.
        if "order_id" not in slots:
            m = re.search(r"\b(?:order|package|parcel|shipment)\b[^.?!]{0,30}?\b(\d{5,12})\b", text, re.I)
            if m:
                slots["order_id"] = m.group(1)

        products = [p for p in self.product_catalog if p in text.lower()]
        if products and "product_name" not in slots:
            slots["product_name"] = max(products, key=len)
        return ExtractedEntities(slots=slots, products=products)


def slot_inventory(instructions) -> dict[str, int]:
    """Count gold placeholder slots across the corpus (for Lex slot design)."""
    counts: dict[str, int] = {}
    for text in instructions:
        for raw in extract_placeholders(text):
            slot = PLACEHOLDER_TO_SLOT.get(raw.lower(), re.sub(r"\W+", "_", raw.lower()))
            counts[slot] = counts.get(slot, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
