"""Generate a small Bitext-shaped CSV so the pipeline can be smoke-tested
without downloading the real dataset from Kaggle."""
import csv, itertools, random, sys

random.seed(7)

SPEC = {
    ("ORDER", "track_order"): (
        ["where is my order {{Order Number}}", "track order {{Order Number}}",
         "i want to check the status of order {{Order Number}}", "has my parcel shipped yet",
         "can you tell me where my package is", "status of my delivery please"],
        "You can track your order in the Orders section of your account; a live tracking link is also emailed at dispatch.",
    ),
    ("ORDER", "cancel_order"): (
        ["i need to cancel order {{Order Number}}", "how do i cancel my order",
         "please cancel the purchase i just made", "i want to call off order {{Order Number}}",
         "can i stop my order from shipping", "cancel my recent purchase"],
        "Orders can be cancelled from Orders > Cancel within 60 minutes of purchase; after that contact support for a return.",
    ),
    ("REFUND", "get_refund"): (
        ["i want a refund for order {{Order Number}}", "how do i get my money back",
         "when will my refund of {{Refund Amount}} arrive", "i was charged but never received the item",
         "requesting reimbursement for my purchase", "refund status please"],
        "Refunds are issued to the original payment method within 5-7 business days of the returned item being received.",
    ),
    ("PAYMENT", "payment_issue"): (
        ["my payment failed", "the card was declined at checkout",
         "i was charged twice for order {{Order Number}}", "why is my payment not going through",
         "problem paying with my credit card", "double charge on my account"],
        "Failed payments are usually an issuer block or an address mismatch; retry with a different method or contact your bank.",
    ),
    ("ACCOUNT", "recover_password"): (
        ["i forgot my password", "how do i reset my password",
         "cannot log into my account", "i need to recover access to my account",
         "locked out of my profile", "password reset email not arriving"],
        "Use Forgot Password on the sign-in page; the reset link is valid for 30 minutes and may land in spam.",
    ),
    ("ACCOUNT", "create_account"): (
        ["i want to open an account", "how do i register",
         "help me create a new profile", "sign me up for an account",
         "what do i need to open an account", "registration help"],
        "Create an account from Sign Up with an email address; you will receive a verification link to activate it.",
    ),
    ("SUBSCRIPTION", "manage_newsletter"): (
        ["unsubscribe me from the newsletter", "stop sending me marketing emails",
         "how do i subscribe to your newsletter", "i want to opt out of emails",
         "change my email preferences", "too many promotional emails"],
        "Email preferences live under Account > Communications, and every newsletter carries a one-click unsubscribe link.",
    ),
    ("DELIVERY", "delivery_options"): (
        ["what shipping options do you offer", "how much is express delivery",
         "do you deliver to {{Delivery City}}", "when will my order arrive",
         "is next day shipping available", "delivery times to my area"],
        "Standard delivery is 3-5 business days, express is next business day; options and costs are shown at checkout.",
    ),
}

rows = []
for (category, intent), (templates, response) in SPEC.items():
    for t in itertools.islice(itertools.cycle(templates), 140):
        prefix = random.choice(["", "hi ", "hello, ", "please ", "hey there ", "i need help, "])
        suffix = random.choice(["", "?", " please", " thanks", " asap", "!"])
        rows.append({
            "flags": random.choice(["B", "BQ", "BM", "BL"]),
            "instruction": f"{prefix}{t}{suffix}",
            "category": category,
            "intent": intent,
            "response": response,
        })

random.shuffle(rows)
out = sys.argv[1] if len(sys.argv) > 1 else "data/sample_bitext.csv"
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["flags", "instruction", "category", "intent", "response"])
    w.writeheader()
    w.writerows(rows)
print(f"wrote {len(rows)} rows to {out}")
