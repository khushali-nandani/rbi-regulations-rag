"""Prompt-injection screening for user questions, retrieved excerpts and model output.

Rule-based on purpose: deterministic, free, unit-testable. It is one layer; the system prompt
(treat <question>/<excerpts> as data) is the second, and citation checking the third.
"""
from __future__ import annotations

import re

_P = re.I
INJECTION_PATTERNS = [
    re.compile(r"\b(ignore|disregard|forget|override|bypass)\b[^.?!\n]{0,40}\b(instructions?|rules?|prompt|guidelines?|polic(y|ies)|restrictions?)\b", _P),
    re.compile(r"\b(reveal|print|show|display|repeat|output|leak|expose|dump|tell me)\b[^.?!\n]{0,30}\b(system prompt|your prompt|your instructions|api[ _-]?key|secret|password|credentials?|hidden)\b", _P),
    re.compile(r"\b(you are now|from now on,? you|pretend (to be|you are)|act as (if you|an? (ai|assistant|chatbot|dan|developer|admin|hacker|unrestricted)))\b", _P),
    re.compile(r"\b(developer mode|jailbreak|do anything now)\b", _P),
    re.compile(r"</?\s*(system|excerpts?|question|instructions?)\s*>", _P),  # tag-spoofing
    re.compile(r"\b(new|updated) (system )?instructions?\s*:", _P),
    re.compile(r"sk-ant-|ANTHROPIC_API_KEY", _P),
]
LEAK_PATTERNS = [re.compile(r"sk-ant-|ANTHROPIC_API_KEY", _P),
                 re.compile(r"ONLY the numbered excerpts provided inside", _P)]  # system prompt text

MAX_QUESTION_CHARS = 1000
_SENT = re.compile(r"(?<=[.?!])\s+|\n+")


def is_injection(text: str) -> bool:
    return any(p.search(text) for p in INJECTION_PATTERNS)


def strip_injection(question: str) -> tuple[str, bool]:
    """Drop sentences that look like instructions to the model. Returns (clean, found_any)."""
    kept, found = [], False
    for s in _SENT.split(question):
        if not s.strip():
            continue
        if is_injection(s):
            found = True
        else:
            kept.append(s.strip())
    return " ".join(kept), found


def leaks_secret(answer: str) -> bool:
    return any(p.search(answer) for p in LEAK_PATTERNS)
