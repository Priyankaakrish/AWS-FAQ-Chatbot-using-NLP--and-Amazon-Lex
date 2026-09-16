"""Normalise a Bitext CSV into the schema this project expects.

The free Bitext sample has `flags, utterance, category, intent` and carries no
answers. The FAQ answers therefore have to come from somewhere else -- which is
how it works in production anyway: support answers are authored by the business
and version-controlled, not derived from the utterance corpus. The model learns
which question is being asked; the company decides what the answer is.

Creates data/faq_answers.csv (one editable row per intent). Edit it and re-run;
your edits are preserved.
"""
import argparse, os, sys
import pandas as pd

UTT = ("instruction", "utterance", "question", "text", "query")
RESP = ("response", "answer", "reply", "output")

ANSWERS = {
 "cancel_order": "You can cancel an order from Orders > Cancel within 60 minutes of purchase. After that the order has entered fulfilment, so please request a return once it arrives.",
 "change_order": "Order changes are possible while the status is still Processing: open Orders, select the order and choose Edit. Once it ships, the only option is a return and re-order.",
 "place_order": "Add items to your basket and choose Checkout. You can pay as a guest, but an account lets you track the order and re-order faster.",
 "track_order": "Track your order under Orders in your account. A live tracking link is emailed at dispatch and usually activates within 24 hours.",
 "delivery_options": "Standard delivery takes 3-5 business days and express arrives the next business day. Options and costs are shown at checkout for your address.",
 "delivery_period": "Most orders arrive within 3-5 business days of dispatch. Your confirmation email lists the estimated delivery date for your address.",
 "set_up_shipping_address": "Add a shipping address under Account > Addresses, or enter one at checkout. You can save several and choose a default.",
 "change_shipping_address": "You can change the shipping address until the order ships, from Orders > Edit. After dispatch the carrier may still allow a redirect via your tracking link.",
 "create_account": "Choose Sign Up and register with your email address. You will receive a verification link to activate the account.",
 "delete_account": "Account deletion is under Account > Privacy > Delete Account. This is permanent and removes your order history, so download any invoices first.",
 "edit_account": "Update your name, email, password and preferences under Account > Profile. Changing your email requires confirming the new address.",
 "switch_account": "Sign out from the profile menu and sign back in with the other account's credentials. Accounts cannot be merged.",
 "recover_password": "Use Forgot Password on the sign-in page. The reset link is valid for 30 minutes and sometimes lands in spam.",
 "registration_problems": "If registration fails, check the email is not already in use and that the verification link has not expired. A new link resolves most cases.",
 "check_refund_policy": "Items can be returned within 30 days of delivery in original condition. Refunds go to the original payment method once the return is received.",
 "get_refund": "Start a return from Orders > Return. Once we receive the item, the refund is issued to your original payment method within 5-7 business days.",
 "track_refund": "Refund status appears under Orders > Returns. After we process it, your bank may take a further 3-5 business days to show the credit.",
 "check_invoice": "Your invoices are listed under Account > Billing, including invoice {{Invoice Number}} for each completed order.",
 "get_invoice": "Download any invoice as a PDF from Account > Billing. Invoices are also attached to the order confirmation email.",
 "check_payment_methods": "We accept major credit and debit cards, PayPal and bank transfer. Available methods are shown at checkout and vary by country.",
 "payment_issue": "Declined payments are usually an issuer block or a billing address mismatch. Confirm the details, try another method, or contact your bank.",
 "check_cancellation_fee": "Cancelling within the 60-minute window is free. After fulfilment begins, return shipping may be deducted from the refund.",
 "contact_customer_service": "You can reach support by email, live chat, or phone on {{Customer Support Phone Number}}. Live chat is usually fastest.",
 "contact_human_agent": "I can hand you to a human agent now. Live chat is available during business hours; out of hours we reply by email within one working day.",
 "complaint": "I am sorry about this. Please file the details under Help > Complaints and a specialist will review and respond within two business days.",
 "review": "You can leave a review from the product page or the link in your delivery confirmation email. Reviews appear after a short moderation check.",
 "newsletter_subscription": "Manage email preferences under Account > Communications. Every newsletter also carries a one-click unsubscribe link.",
}

def find(cols, names):
    low = {c.lower().strip(): c for c in cols}
    return next((low[n] for n in names if n in low), None)

def main():
    for s in (sys.stdout, sys.stderr):
        try: s.reconfigure(encoding="utf-8", errors="replace")
        except Exception: pass
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", dest="dst", default="data/bitext_ready.csv")
    p.add_argument("--answers", default="data/faq_answers.csv")
    a = p.parse_args()

    df = pd.read_csv(a.src)
    print(f"Read {len(df):,} rows with columns: {list(df.columns)}")
    utt, resp = find(df.columns, UTT), find(df.columns, RESP)
    intent = find(df.columns, ("intent", "label"))
    cat = find(df.columns, ("category", "topic"))
    if not utt or not intent:
        raise SystemExit(f"Could not find utterance/intent columns in {list(df.columns)}")

    out = pd.DataFrame({
        "flags": df["flags"] if "flags" in df.columns else "",
        "instruction": df[utt].astype(str),
        "category": (df[cat] if cat else "GENERAL"),
        "intent": df[intent].astype(str).str.strip(),
    })
    out["category"] = out["category"].astype(str).str.strip().str.upper()

    if resp:
        out["response"] = df[resp].astype(str)
        print(f"Using the dataset's own '{resp}' column as FAQ answers.")
    else:
        bank = pd.DataFrame(columns=["intent", "response"])
        if os.path.exists(a.answers):
            bank = pd.read_csv(a.answers)
            bank.columns = [c.lower().strip() for c in bank.columns]
        known = set(bank["intent"]) if len(bank) else set()
        new = [{"intent": i,
                "response": ANSWERS.get(i, f"Our support team can help with {i.replace('_',' ')}. Please see the Help Centre for the current policy.")}
               for i in sorted(out["intent"].unique()) if i not in known]
        if new:
            bank = pd.concat([bank, pd.DataFrame(new)], ignore_index=True)
        bank.to_csv(a.answers, index=False, encoding="utf-8")
        out = out.merge(bank, on="intent", how="left")
        print(f"No answer column found (intent-only sample).")
        print(f"Wrote an editable answer bank for {len(bank)} intents to {a.answers}.")
        print("EDIT THAT FILE to match your real policies, then re-run this script.")

    out.to_csv(a.dst, index=False, encoding="utf-8")
    print(f"\nWrote {len(out):,} rows | {out['intent'].nunique()} intents | {out['category'].nunique()} categories -> {a.dst}")
    print(f"Next:  python -m src.train --data {a.dst} --out artifacts")

if __name__ == "__main__":
    main()