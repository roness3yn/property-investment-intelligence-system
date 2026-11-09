"""Output guardrail: checks a draft answer before the user sees it (README criterion 7).

1. Groundedness. Every number in the answer is matched against the numbers in the tool results
   of this turn. A number is
     * grounded       if it matches a result the same sentence (or line, or paragraph) cites;
     * uncited        if it matches some result of this turn, but the answer does not cite it;
     * ungrounded     if no tool result and no user message contains it.
   Small counts (0-10), dates and the evidence references themselves are not claims.
   Rounding is allowed: "342" matches 342.37, "22,300" matches 22,344, "58%" matches 0.577.
2. Citations. Every cited reference [E7] must exist.
3. Policy. No buy instructions, no guaranteed returns, and no ROI/yield figures unless the
   roi_calculator ran on the user's own numbers.

The report is shown in the chat under each answer. When the check fails, the orchestrator asks
a reviser model to fix the draft once; whatever still fails is marked ⚠️ in the answer.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from agents.evidence import NUMBER_RE, EvidenceStore, numbers_in_text, parse_number

REF_RE = re.compile(r"\[(E\d+(?:\s*,\s*E\d+)*)\]")
DATE_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b\d{1,2}[./]\d{1,2}[./]\d{2,4}\b")
EVID_RE = re.compile(r"`[^`]*`|\b(?:opp|pred|nbh|rev|avail|roi|opp-listing|reg|rules|card|seg|lst):[\w:./\-+]+")

POLICY_RULES = [
    ("buy_instruction", r"\b(you should|i recommend|we recommend) (buy|purchase|invest)|\bbuy (it|this|now)\b"
                        r"|מומלץ לקנות|כדאי לקנות|כדאי להשקיע|קנה עכשיו|לך על זה|תקנה את"),
    ("guarantee", r"\bguarantee[ds]?\b|\brisk[- ]free\b|השקעה בטוחה|ללא סיכון|מובטח|מובטחת|בטוח ירוויח"),
]
ROI_WORDS = r"\bROI\b|return on investment|\byield\b|תשואה|החזר על ההשקעה"
NEGATION = r"(not|no|isn't|is not|without|לא|אינו|אינה|אין|ללא|איננו|בלי)\b"


@dataclass
class Claim:
    value: float
    text: str
    cited: list[str]
    status: str                     # grounded | uncited | ungrounded
    matched_ref: str | None = None


@dataclass
class VerificationReport:
    claims: list[Claim] = field(default_factory=list)
    unknown_refs: list[str] = field(default_factory=list)
    violations: list[dict] = field(default_factory=list)

    @property
    def n_claims(self) -> int:
        return len(self.claims)

    @property
    def ungrounded(self) -> list[Claim]:
        return [c for c in self.claims if c.status == "ungrounded"]

    @property
    def groundedness(self) -> float | None:
        if not self.claims:
            return None
        return sum(c.status != "ungrounded" for c in self.claims) / len(self.claims)

    @property
    def citation_rate(self) -> float | None:
        if not self.claims:
            return None
        return sum(c.status == "grounded" for c in self.claims) / len(self.claims)

    @property
    def passed(self) -> bool:
        return not self.ungrounded and not self.unknown_refs and not self.violations

    def as_dict(self) -> dict:
        return {"passed": self.passed, "claims": self.n_claims,
                "groundedness": self.groundedness, "citation_rate": self.citation_rate,
                "ungrounded": [{"value": c.value, "text": c.text} for c in self.ungrounded],
                "uncited": [{"value": c.value, "text": c.text} for c in self.claims if c.status == "uncited"],
                "unknown_refs": self.unknown_refs, "violations": self.violations}

    def feedback(self) -> str:
        """Instructions for the reviser."""
        lines = []
        for c in self.ungrounded:
            lines.append(f"- The number {c.text!r} is not in any tool result. Remove it, or replace it "
                         "with the exact value from a tool result and cite its reference.")
        for c in self.claims:
            if c.status == "uncited":
                lines.append(f"- The number {c.text!r} needs a citation: add the reference [{c.matched_ref}].")
        for r in self.unknown_refs:
            lines.append(f"- [{r}] does not exist. Cite only references returned by the tools.")
        for v in self.violations:
            lines.append(f"- Policy ({v['rule']}): {v['detail']}")
        return "\n".join(lines)


def _round_unit(token: str) -> float:
    t = token.replace(",", "").replace("−", "-").lstrip("-")
    if "." in t:
        return 10 ** -len(t.split(".")[1])
    stripped = t.rstrip("0")
    zeros = len(t) - len(stripped) if stripped else 0
    return float(10 ** min(zeros, 3))


def _matches(value: float, token: str, candidates: set[float]) -> float | None:
    half = _round_unit(token) / 2 + 1e-9
    for e in candidates:
        for form in (e, e * 100):                      # 0.577 can be written as 57.7%
            if math.isfinite(form) and abs(abs(value) - abs(form)) <= max(half, 0.005 * abs(form)):
                return e
    return None


def _segments(text: str):
    """Yield (sentence, line, context) triples. The context is the paragraph; for the rows of
    a markdown table it also includes the nearest earlier paragraph that cites evidence, since
    a table is usually cited in its caption."""
    last_cited = ""
    for para in re.split(r"\n\s*\n", text):
        is_table = sum(l.strip().startswith("|") for l in para.split("\n")) >= 2
        context = para + ("\n" + last_cited if is_table else "")
        for line in para.split("\n"):
            parts = re.split(r"(?<=[.!?؟])\s+(?=\S)", line) if "|" not in line else [line]
            for sent in parts:
                yield sent, line, context
        if REF_RE.search(para):
            last_cited = para


def _refs(text: str) -> list[str]:
    out = []
    for m in REF_RE.finditer(text):
        out += [r.strip() for r in m.group(1).split(",")]
    return out


def verify(answer: str, store: EvidenceStore, user_text: str = "",
           small_int_max: int = 10) -> VerificationReport:
    report = VerificationReport()
    user_nums = set(numbers_in_text(user_text))
    all_nums = store.all_numbers()
    known = set(store.records)
    report.unknown_refs = sorted({r for r in _refs(answer) if r not in known},
                                 key=lambda r: int(r[1:]))

    for sent, line, para in _segments(answer):
        clean = REF_RE.sub(" ", EVID_RE.sub(" ", DATE_RE.sub(" ", sent)))
        cited = _refs(sent) or _refs(line) or _refs(para)
        cited = [r for r in cited if r in known]
        for m in NUMBER_RE.finditer(clean):
            token = m.group(0)
            v = parse_number(token)
            if v is None:
                continue
            if float(v).is_integer() and abs(v) <= small_int_max:
                continue                                # counts, ranks, bedrooms
            if 1990 <= v <= 2100 and float(v).is_integer():
                continue                                # years
            if _matches(v, token, user_nums) is not None:
                continue                                # the user's own numbers are not claims
            hit = _matches(v, token, store.all_numbers(cited)) if cited else None
            if hit is not None:
                report.claims.append(Claim(v, token, cited, "grounded"))
                continue
            if _matches(v, token, all_nums) is not None:
                ref = next((r.ref for r in store.records.values()
                            if _matches(v, token, r.numbers) is not None), None)
                report.claims.append(Claim(v, token, cited, "uncited", ref))
            else:
                report.claims.append(Claim(v, token, cited, "ungrounded"))

    for rule, rx in POLICY_RULES:
        m = re.search(rx, answer, re.IGNORECASE)
        if m:
            report.violations.append({"rule": rule, "detail": f"remove {m.group(0)!r}: the system "
                                      "supports investigation and never tells the user to buy or "
                                      "promises a return"})
    if not store.used_tool("roi_calculator"):
        for m in re.finditer(ROI_WORDS, answer, re.IGNORECASE):
            before = answer[max(0, m.start() - 30):m.start()]
            window = answer[m.start():m.end() + 40]
            if re.search(NEGATION, before, re.IGNORECASE):
                continue                                # "this is not an ROI" is fine
            if numbers_in_text(window):
                report.violations.append({"rule": "roi_without_inputs", "detail":
                    f"{window.strip()!r}: an ROI or yield needs a purchase price and costs from the "
                    "user and the roi_calculator tool; the score is rental revenue potential only"})
                break
    return report


def mark_ungrounded(answer: str, report: VerificationReport) -> str:
    """Last resort after revision: put ⚠️ after each number that is still ungrounded."""
    for c in report.ungrounded:
        answer = re.sub(rf"(?<![\w.]){re.escape(c.text)}(?![\d])", f"{c.text}⚠️", answer, count=1)
    return answer
