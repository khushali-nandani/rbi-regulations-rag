from __future__ import annotations

from dataclasses import dataclass, field

from .injection import MAX_QUESTION_CHARS, leaks_secret, strip_injection
from .pii import mask_pii


@dataclass
class InputCheck:
    clean: str  # PII-masked, injection-stripped question: the ONLY form that may travel onward
    masked: str = ""  # PII-masked but otherwise original (safe to log)
    pii_types: dict[str, int] = field(default_factory=dict)
    injection: bool = False
    blocked: str | None = None  # None | "too_long" | "injection_only" | "empty"


def check_input(question: str) -> InputCheck:
    if len(question) > MAX_QUESTION_CHARS:
        return InputCheck(clean="", masked="[question too long]", blocked="too_long")
    masked, pii = mask_pii(question)
    clean, found = strip_injection(masked)
    if found and len(clean.split()) < 3:  # nothing legitimate left after removing the injection
        return InputCheck(clean="", masked=masked, pii_types=pii, injection=True,
                          blocked="injection_only")
    if not clean.strip():
        return InputCheck(clean="", masked=masked, pii_types=pii, blocked="empty")
    return InputCheck(clean=clean, masked=masked, pii_types=pii, injection=found)


def check_output(answer: str) -> tuple[str, bool]:
    """Returns (safe_answer, withheld). Masks PII; withholds anything that leaks secrets/prompt."""
    if leaks_secret(answer):
        return "", True
    masked, _ = mask_pii(answer, keep_email_domains=("rbi.org.in",))
    return masked, False
