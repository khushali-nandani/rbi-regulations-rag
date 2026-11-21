"""PII masking for Indian identifiers. Runs BEFORE retrieval, the LLM and any logging.

Detected values are replaced by typed placeholders; only the *types* are reported, never values.
Order matters: card (Luhn-checked) -> +91 phone -> Aadhaar -> mobile -> PAN -> email.
"""
from __future__ import annotations

import re
from collections import Counter

CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
AADHAAR = re.compile(r"(?<!\d)[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}(?!\d)")
PHONE_INTL = re.compile(r"(?<!\d)\+91[ -]?[6-9]\d{9}(?!\d)")  # explicit +91 beats the 12-digit Aadhaar shape
MOBILE = re.compile(r"(?<!\d)(?:\+?91[ -]?)?[6-9]\d{9}(?!\d)")
PAN = re.compile(r"\b[A-Za-z]{5}\d{4}[A-Za-z]\b")
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def luhn_ok(digits: str) -> bool:
    d = [int(c) for c in digits][::-1]
    total = sum(d[0::2]) + sum(sum(divmod(2 * x, 10)) for x in d[1::2])
    return total % 10 == 0


def mask_pii(text: str, keep_email_domains: tuple[str, ...] = ()) -> tuple[str, dict[str, int]]:
    found: Counter = Counter()

    def card(m: re.Match) -> str:
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn_ok(digits):
            found["CARD"] += 1
            return "[CARD]"
        return m.group(0)

    def sub(label: str):
        def _f(m: re.Match) -> str:
            found[label] += 1
            return f"[{label}]"
        return _f

    text = CARD.sub(card, text)
    text = PHONE_INTL.sub(sub("PHONE"), text)
    text = AADHAAR.sub(sub("AADHAAR"), text)
    text = MOBILE.sub(sub("PHONE"), text)
    text = PAN.sub(sub("PAN"), text)
    def email(m: re.Match) -> str:
        if m.group(0).lower().endswith(tuple(keep_email_domains)):  # e.g. official crpc@rbi.org.in
            return m.group(0)
        found["EMAIL"] += 1
        return "[EMAIL]"

    text = EMAIL.sub(email, text)
    return text, dict(found)
