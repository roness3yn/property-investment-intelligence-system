"""The orchestrator: one user message in, one checked answer out.

    user message
        -> input guardrail          (deterministic: injection, size)
        -> Router agent             (fast model, forced tool call: intent + city)
        -> specialist agent         Opportunity Investigator | Price Analyst | Data Enrichment
                                    (each with its own tools; the first two can delegate to the
                                    Enrichment agent as a tool)
        -> output guardrail         (deterministic: groundedness of every number, citations, policy)
        -> Reviser agent            (only if the check failed; then checked again)
        -> answer + trace + evidence + verification report

Without an API key, or if the language model fails, the orchestrator falls back to the
deterministic path the charter requires: the opportunity table and the data status, with an
explicit "agent unavailable" status. The investor is never left without the analytical output.
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field

from agents import prompts
from agents.config import AgentConfig
from agents.evidence import EvidenceStore
from agents.guardrails import ToolGuard, TurnState, check_input, mark_ungrounded, verify
from agents.guardrails.output_guard import VerificationReport
from agents.llm import AgentResult, AgentSpec, AnthropicLLM, ask_once, make_delegate, run_agent
from agents.tools import ToolContext, build_registry

INTENTS = ["investigate", "price", "data", "help", "out_of_scope"]
ROUTE_TOOL = {
    "name": "route_request",
    "description": "Route the user's message to the right specialist.",
    "input_schema": {"type": "object", "properties": {
        "intent": {"type": "string", "enum": INTENTS},
        "city": {"type": ["string", "null"]},
        "reason": {"type": "string", "description": "One short sentence."}},
        "required": ["intent", "city", "reason"]},
}
AGENT_TOOLS = {   # what each specialist may use; tools whose data is missing are removed at start-up
    "investigator": ["rank_opportunities", "segment_profile", "listing_profile", "model_card",
                     "neighbourhood_stats", "review_signals", "predict_listing_price"],
    "price_analyst": ["predict_listing_price", "rank_opportunities", "model_card", "segment_profile",
                      "neighbourhood_stats", "roi_calculator"],
    "enrichment": ["data_availability", "neighbourhood_stats", "segment_profile", "model_card",
                   "review_signals", "web_search"],
}
TITLES = {"router": "נתב", "investigator": "חוקר הזדמנויות", "price_analyst": "אנליסט מחירים",
          "enrichment": "סוכן העשרת נתונים", "verifier": "מאמת ראיות", "reviser": "מתקן ראיות",
          "fallback": "מצב ללא סוכן"}

HELP_HE = """אני מערכת סוכנים לחקר הזדמנויות השקעה בהשכרה לטווח קצר, על נתוני Inside Airbnb.

**מה אפשר לשאול:**
- **דירוג וחקירה**: "מהן 5 ההזדמנויות המובילות באמסטרדם ולמה?", "מה הסיכונים בהזדמנות הראשונה?"
- **מחיר ו-what-if**: "כמה תכניס דירת 2 חדרי שינה ב-De Pijp ל-4 אורחים?", "האם listing 28871 מתומחר נמוך?"
- **תשואה**: "אם אקנה ב-450,000 EUR עם עלויות 9,000 בשנה, מה התשואה?" (רק עם המספרים שלך)
- **נתונים והקשר**: "אילו ערים יש בנתונים?", "מה המצב בברצלונה?", "מה הרגולציה על Airbnb באמסטרדם?"

**איך זה עובד:** נתב מעביר את השאלה לסוכן מתמחה (חוקר הזדמנויות, אנליסט מחירים או סוכן העשרה). הסוכן קורא לכלים קריאה-בלבד, וכל מספר בתשובה נבדק אוטומטית מול תוצאות הכלים ומצוטט [E#].

**מה הציון אומר:** הכנסה שנתית שמרנית מהשכרה (לילות מוזמנים × מחיר צפוי × ודאות). זו **לא** תשואה על השקעה ולא המלצת קנייה."""
OUT_OF_SCOPE_HE = ("השאלה מחוץ לתחום של המערכת. אני עוזר לחקור הזדמנויות השקעה בהשכרה לטווח קצר: "
                   "דירוג הזדמנויות, מחירים צפויים, נתוני שכונות וביקורות, והקשר רגולטורי. "
                   "למשל: \"מהן ההזדמנויות המובילות באמסטרדם?\"")


@dataclass
class TurnResult:
    turn_id: str
    question: str
    answer: str
    intent: str
    agent: str
    city: str | None
    events: list
    evidence: list
    verification: dict | None
    usage: dict
    elapsed_s: float
    mode: str                               # agent | fallback | blocked | canned
    route_reason: str | None = None
    tables: dict = field(default_factory=dict)


class Orchestrator:
    def __init__(self, config: AgentConfig | None = None, llm=None, api_key: str | None = None):
        self.config = config or AgentConfig()
        self.ctx = ToolContext(self.config)
        self.registry = build_registry(self.ctx)
        self.guard = ToolGuard(self.registry, self.config)
        self.llm = llm
        if self.llm is None and (api_key or self.config.api_key_present):
            try:
                self.llm = AnthropicLLM(api_key)
            except Exception:
                self.llm = None
        self.capabilities = self.ctx.capabilities()
        self.deployment = self._deployment_note()
        self.specs = self._build_specs()

    # ------------------------------------------------------------------ agents
    def available_tools(self, agent: str) -> list[str]:
        return [t for t in AGENT_TOOLS[agent] if self.capabilities.get(t) is None]

    def _deployment_note(self) -> str:
        """What this deployment can and cannot answer, appended to every specialist's prompt, so
        the agents plan with the tools that exist and state what is missing."""
        st = self.ctx.status()
        lines = ["## This deployment (generated at start-up from the files present)"]
        for city, t in st["opportunity_tables"].items():
            lines.append(f"- Opportunity table: {city}, {t['ranked']} ranked opportunities, snapshot "
                         f"{t['snapshot_date']}, currency {t['currency']}, label {t['label']}"
                         + (" (test listings not included; the model's test run was exploratory)"
                            if t["label"] == "PROVISIONAL" else "")
                         + (" (SYNTHETIC demo data: numbers are not real)" if t["label"] == "SYNTHETIC" else ""))
        if st["price_models"]:
            lines.append(f"- Price model: {', '.join(st['price_models'])}")
        missing = {t: why for t, why in self.capabilities.items() if why}
        for t, why in missing.items():
            lines.append(f"- Not available: `{t}`: {why}.")
        if "review_signals" in missing:
            lines.append("- Guest review TEXT was not analysed. For guest experience use the structured "
                         "signals (review scores, recent review counts) from segment_profile and "
                         "listing_profile, and say that review text was not analysed.")
        return "\n".join(lines)

    def _build_specs(self) -> dict[str, AgentSpec]:
        specs = {}
        for name in ["investigator", "price_analyst", "enrichment"]:
            system, version = prompts.system_prompt(name)
            specs[name] = AgentSpec(name=name, title_he=TITLES[name],
                                    system=f"{system}\n\n{self.deployment}",
                                    prompt_version=version, tools=self.available_tools(name))
        delegate = make_delegate(
            "ask_data_enrichment_agent",
            "Ask the Data Enrichment Agent for data you do not have: which cities and data exist, "
            "statistics for a city without a model, or outside context (regulation, licences, "
            "taxes) from the web. It answers with findings that cite evidence references.",
            self._run_enrichment_for)
        for name in ["investigator", "price_analyst"]:
            specs[name].delegates = {"ask_data_enrichment_agent": delegate}
        return specs

    def _run_enrichment_for(self, question: str, city: str | None, state: TurnState) -> AgentResult:
        task = f"[Request from another agent]\nCity: {city or 'not specified'}\nQuestion: {question}"
        return run_agent(self.specs["enrichment"], task, llm=self.llm, guard=self.guard,
                         state=state, config=self.config)

    # ------------------------------------------------------------------ routing
    def route(self, text: str, history: list[dict], state: TurnState) -> dict:
        if self.llm is not None:
            system, version = prompts.system_prompt("router", with_common=False)
            recent = "\n".join(f"{m['role']}: {str(m['content'])[:400]}" for m in history[-4:])
            user = (f"Recent conversation:\n{recent or '(none)'}\n\nLatest message:\n{text}")
            state.emit("agent_start", "router", title=TITLES["router"], prompt_version=version,
                       model=self.config.router_model, tools=["route_request"])
            try:
                out = ask_once(self.llm, model=self.config.router_model, system=system, user=user,
                               max_tokens=300, state=state, tool=ROUTE_TOOL)
                if out.get("intent") in INTENTS:
                    state.emit("route", "router", intent=out["intent"], city=out.get("city"),
                               reason=out.get("reason"))
                    return out
            except Exception as e:
                state.emit("error", "router", error=f"{type(e).__name__}: {e}")
        out = heuristic_route(text)
        state.emit("route", "router", intent=out["intent"], city=out.get("city"),
                   reason=out["reason"], heuristic=True)
        return out

    # ------------------------------------------------------------------ main entry
    def handle(self, message: str, history: list[dict] | None = None, on_event=None) -> TurnResult:
        history = list(history or [])[-2 * self.config.history_turns:]
        turn_id = time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]
        state = TurnState(store=EvidenceStore(self.config.evidence_log, turn_id),
                          user_text="\n".join([m["content"] for m in history if m["role"] == "user"] + [message]),
                          on_event=on_event)
        t0 = time.time()

        def result(answer, intent, agent, mode, city=None, verification=None, reason=None, tables=None):
            return TurnResult(turn_id, message, answer, intent, agent, city, state.events,
                              [{**r.summary(), "result": r.result} for r in state.store.records.values()],
                              verification, state.usage, round(time.time() - t0, 2), mode, reason,
                              tables or {})

        check = check_input(message, self.config.max_input_chars)
        if not check.allowed:
            state.emit("guardrail", "input", guard="input", reason=check.reason)
            return result(check.message_he, "blocked", "input_guardrail", "blocked")

        route = self.route(check.text, history, state)
        intent, city = route["intent"], route.get("city")
        if intent == "help":
            return result(HELP_HE, intent, "router", "canned", city, reason=route.get("reason"))
        if intent == "out_of_scope":
            return result(OUT_OF_SCOPE_HE, intent, "router", "canned", city, reason=route.get("reason"))
        if self.llm is None:
            return self.fallback(check.text, intent, city, state, result, "no API key")

        agent = {"investigate": "investigator", "price": "price_analyst", "data": "enrichment"}[intent]
        task = check.text if not city else f"{check.text}\n\n(City in context: {city})"
        try:
            res = run_agent(self.specs[agent], task, llm=self.llm, guard=self.guard, state=state,
                            config=self.config, history=history)
            answer, report = self.verify_and_revise(res.text, state)
        except Exception as e:
            state.emit("error", agent, error=f"{type(e).__name__}: {e}")
            return self.fallback(check.text, intent, city, state, result, f"{type(e).__name__}: {e}")
        return result(answer, intent, agent, "agent", city, report.as_dict(), route.get("reason"))

    # ------------------------------------------------------------------ verification
    def verify_and_revise(self, draft: str, state: TurnState) -> tuple[str, VerificationReport]:
        report = verify(draft, state.store, state.user_text)
        state.emit("verify", "verifier", title=TITLES["verifier"], **_short(report))
        if report.passed or not draft:
            return draft, report
        system, version = prompts.system_prompt("reviser", with_common=False)
        evidence = "\n".join(
            f"[{r.ref}] {r.tool} {r.status}: " + _compact(r.result) for r in state.store.records.values())
        user = (f"## Evidence of this turn\n{evidence[:12000]}\n\n## Problems found\n{report.feedback()}"
                f"\n\n## Draft answer\n{draft}")
        state.emit("agent_start", "reviser", title=TITLES["reviser"], prompt_version=version,
                   model=self.config.router_model, tools=[])
        try:
            revised = ask_once(self.llm, model=self.config.router_model, system=system, user=user,
                               max_tokens=self.config.max_tokens, state=state).get("text") or draft
        except Exception as e:
            state.emit("error", "reviser", error=f"{type(e).__name__}: {e}")
            revised = draft
        report2 = verify(revised, state.store, state.user_text)
        state.emit("verify", "verifier", title=TITLES["verifier"], after_revision=True, **_short(report2))
        if not report2.passed:
            revised = mark_ungrounded(revised, report2)
            if report2.ungrounded or report2.violations:
                revised += ("\n\n> ⚠️ **בדיקת הראיות:** חלק מהתשובה לא אומת מול תוצאות הכלים "
                            "(מסומן ⚠️). התייחס אליו בזהירות.")
        return revised, report2

    # ------------------------------------------------------------------ deterministic fallback
    def fallback(self, text, intent, city, state, result, why):
        state.emit("guardrail", "fallback", guard="agent_unavailable", reason=why)
        allowed = {"rank_opportunities", "data_availability"}
        top, ref = self.guard.execute("fallback", allowed, "rank_opportunities",
                                      {"action": "top", "n": 10, **({"city": city} if city else {})}, state)
        lines = [f"> **מצב ללא סוכן** ({why}). מוצגת הטבלה הדטרמיניסטית, כפי שהאמנה מחייבת.", ""]
        tables = {}
        if top.get("status") == "ok":
            cur = top.get("currency") or ""
            lines += [f"**10 ההזדמנויות המובילות ב-{top.get('city')}** (תווית: {top.get('label')}) [{ref}]", "",
                      f"| דירוג | שכונה | סוג | חדרי שינה | ציון ({cur}/שנה) | מחיר ללילה | לילות | ודאות |",
                      "|---|---|---|---|---|---|---|---|"]
            for r in top["results"]:
                lines.append(f"| {r['rank']} | {r['neighbourhood']} | {r['room_type']} | {r['bedrooms']} | "
                             f"{r['score_conservative_annual_revenue']:,.0f} | {r['expected_nightly_price']:,.0f} | "
                             f"{r['expected_booked_nights']:,.0f} | {r['price_certainty']:.2f} |")
            lines += ["", "הציון הוא הכנסה שנתית שמרנית מהשכרה, לא תשואה על השקעה ולא המלצת קנייה."]
            tables["top"] = top["results"]
        else:
            avail, aref = self.guard.execute("fallback", allowed, "data_availability", {}, state)
            lines += [f"אין טבלת הזדמנויות זמינה: {top.get('status_reason')}.",
                      f"ערים בנתונים: {', '.join(c['city'] for c in avail.get('cities', [])) or 'אין'} [{aref}]"]
        lines += ["", "כדי לקבל חקירה מלאה עם סוכנים, הגדר `ANTHROPIC_API_KEY` (בסרגל הצד או בקובץ `.env`)."]
        answer = "\n".join(lines)
        report = verify(answer, state.store, state.user_text)
        return result(answer, intent, "fallback", "fallback", city, report.as_dict(), why, tables)


def _short(report: VerificationReport) -> dict:
    d = report.as_dict()
    return {"passed": d["passed"], "claims": d["claims"], "groundedness": d["groundedness"],
            "ungrounded": len(d["ungrounded"]), "uncited": len(d["uncited"]),
            "violations": [v["rule"] for v in d["violations"]], "unknown_refs": d["unknown_refs"]}


def _compact(result: dict, limit: int = 1500) -> str:
    import json
    r = {k: v for k, v in result.items() if k not in ("assumptions", "guardrail", "load_errors")}
    return json.dumps(r, ensure_ascii=False, default=str)[:limit]


def heuristic_route(text: str) -> dict:
    """Keyword routing, used without an API key or if the router model fails."""
    t = text.lower()
    city = next((c for c in ["Amsterdam", "Barcelona", "London", "Paris", "Lisbon", "Berlin", "Rome",
                             "Madrid", "Prague", "Vienna", "Athens", "Dublin", "Milan", "Porto"]
                 if c.lower() in t), None)
    city = city or ("Amsterdam" if "אמסטרדם" in text else "Barcelona" if "ברצלונה" in text else None)
    if re.search(r"^(hi|hello|help|שלום|היי|עזרה)\b|what can you do|מה אתה יודע|איך זה עובד", t):
        return {"intent": "help", "city": city, "reason": "keywords: help"}
    if re.search(r"listing|\bprice\b|nightly|per night|מחיר|ללילה|כמה (תכניס|יכניס|עולה)|roi|תשואה|what.?if", t):
        return {"intent": "price", "city": city, "reason": "keywords: price"}
    if re.search(r"data|cities|regulat|licen|tax|נתונים|ערים|רגולציה|רישיון|מס\b|חוק", t):
        return {"intent": "data", "city": city, "reason": "keywords: data"}
    return {"intent": "investigate", "city": city, "reason": "default: investigation"}
