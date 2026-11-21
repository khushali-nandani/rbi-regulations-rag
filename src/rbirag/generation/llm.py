"""Thin Anthropic wrapper: disk cache + cumulative spend guard + usage accounting.

Every call is cached on (model, system, user, params), so re-running an eval never pays twice,
and a hard spend limit stops runaway loops before the $5 budget is touched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import ROOT, settings

# $ per 1M tokens (input, output) - from the Anthropic model table; prompts < 100K tokens.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00)}
CACHE_DIR = ROOT / ".cache/llm"
SPEND_FILE = ROOT / ".cache/llm_spend.json"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    stop_reason: str = "end_turn"
    cached: bool = False
    model: str = ""
    meta: dict = field(default_factory=dict)


def cost_of(model: str, in_tok: int, out_tok: int) -> float:
    pi, po = PRICES.get(model, (2.0, 10.0))  # unknown model: assume Sonnet-class pricing
    return (in_tok * pi + out_tok * po) / 1e6


def total_spend() -> float:
    try:
        return float(json.loads(SPEND_FILE.read_text())["usd"])
    except Exception:
        return 0.0


def _add_spend(usd: float) -> None:
    SPEND_FILE.parent.mkdir(parents=True, exist_ok=True)
    SPEND_FILE.write_text(json.dumps({"usd": total_spend() + usd}))


class LLMStream:
    """Iterate to receive text deltas; `.result` is set once the stream is exhausted."""

    def __init__(self, gen_factory):
        self._gen_factory, self.result = gen_factory, None

    def __iter__(self):
        yield from self._gen_factory(self)


class LLM:
    def __init__(self, client=None):
        self._client = client

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key or None)
        return self._client

    def complete(self, *, system: str, user: str, model: str, max_tokens: int = 700,
                 effort: str = "low") -> LLMResult:
        key = hashlib.sha256(json.dumps([model, system, user, max_tokens, effort]).encode()).hexdigest()
        f = CACHE_DIR / f"{key}.json"
        if f.exists():
            return LLMResult(**{**json.loads(f.read_text()), "cached": True})
        if total_spend() >= settings.max_spend_usd:
            raise BudgetExceeded(f"spent ${total_spend():.2f} >= limit ${settings.max_spend_usd:.2f}")
        resp = self.client.messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
            output_config={"effort": effort},
        )
        text = "".join(b.text for b in resp.content if b.type == "text")
        u = resp.usage
        in_tok = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) \
            + (getattr(u, "cache_read_input_tokens", 0) or 0)
        res = LLMResult(text=text, input_tokens=in_tok, output_tokens=u.output_tokens or 0,
                        cost_usd=cost_of(model, in_tok, u.output_tokens or 0),
                        stop_reason=resp.stop_reason or "end_turn", model=model)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps({k: v for k, v in res.__dict__.items() if k != "cached"}))
        _add_spend(res.cost_usd)
        return res

    def stream(self, *, system: str, user: str, model: str, max_tokens: int = 700,
               effort: str = "low") -> LLMStream:
        key = hashlib.sha256(json.dumps([model, system, user, max_tokens, effort]).encode()).hexdigest()
        f = CACHE_DIR / f"{key}.json"

        def gen(handle: LLMStream):
            if f.exists():  # cache hit: replay in small pieces so the client still sees a stream
                res = LLMResult(**{**json.loads(f.read_text()), "cached": True})
                for i in range(0, len(res.text), 24):
                    yield res.text[i:i + 24]
                handle.result = res
                return
            if total_spend() >= settings.max_spend_usd:
                raise BudgetExceeded(f"spent ${total_spend():.2f} >= limit ${settings.max_spend_usd:.2f}")
            with self.client.messages.stream(
                model=model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": user}], output_config={"effort": effort},
            ) as st:
                for piece in st.text_stream:
                    yield piece
                final = st.get_final_message()
            u = final.usage
            in_tok = (u.input_tokens or 0) + (getattr(u, "cache_creation_input_tokens", 0) or 0) \
                + (getattr(u, "cache_read_input_tokens", 0) or 0)
            text = "".join(b.text for b in final.content if b.type == "text")
            res = LLMResult(text=text, input_tokens=in_tok, output_tokens=u.output_tokens or 0,
                            cost_usd=cost_of(model, in_tok, u.output_tokens or 0),
                            stop_reason=final.stop_reason or "end_turn", model=model)
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            f.write_text(json.dumps({k: v for k, v in res.__dict__.items() if k != "cached"}))
            _add_spend(res.cost_usd)
            handle.result = res

        return LLMStream(gen)
