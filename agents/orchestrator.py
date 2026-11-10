"""The orchestrator: one user message in, one checked answer out.

    user message
        -> input guardrail          (deterministic: injection, size)
        -> Router agent             (fast model, forced tool call: intent + city)
        -> specialist agent         Opportunity Investigator | Price Analyst | Regulation &
                                    Compliance | Data Enrichment (each with its own tools; the
                                    first three can delegate to the Enrichment agent as a tool)
        -> output guardrail         (deterministic: groundedness of every number, citations, policy)
        -> Reviser agent            (only if the check failed; then checked again)
        -> regulation check         (deterministic: every opportunity in the answer that the
                                    agent did not check gets compliance_check, appended)
        -> answer + trace + evidence + verification report

Without an API key, or if the language model fails, the orchestrator falls back to the
deterministic path the charter requires: the opportunity table and the data status, with an
explicit "agent unavailable" status, with the regulation applied. The investor is never left
without the analytical output.
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

INTENTS = ["investigate", "price", "regulation", "data", "help", "out_of_scope"]
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
                     "neighbourhood_stats", "review_signals", "predict_listing_price",
                     "compliance_check", "regulation_rules"],
    "price_analyst": ["predict_listing_price", "rank_opportunities", "model_card", "segment_profile",
                      "neighbourhood_stats", "roi_calculator", "compliance_check", "regulation_rules"],
    "compliance": ["regulation_rules", "compliance_check", "rank_opportunities", "segment_profile"],
    "enrichment": ["data_availability", "neighbourhood_stats", "segment_profile", "model_card",
                   "review_signals", "regulation_rules", "web_search"],
}
SPECIALISTS = ["investigator", "price_analyst", "compliance", "enrichment"]
INTENT_AGENT = {"investigate": "investigator", "price": "price_analyst", "regulation": "compliance",
                "data": "enrichment"}
TITLES = {"router": "נתב", "investigator": "חוקר הזדמנויות", "price_analyst": "אנליסט מחירים",
          "compliance": "סוכן רגולציה וציות", "enrichment": "סוכן העשרת נתונים",
          "verifier": "מאמת ראיות", "reviser": "מתקן ראיות", "regulation_check": "בדיקת רגולציה",
          "fallback": "מצב ללא סוכן"}

HELP_HE = """אני מערכת סוכנים לחקר הזדמנויות השקעה בהשכרה לטווח קצר, על נתוני Inside Airbnb.

**מה אפשר לשאול:**
- **דירוג וחקירה**: "מהן 5 ההזדמנויות המובילות באמסטרדם ולמה?", "מה הסיכונים בהזדמנות הראשונה?"
- **מחיר ו-what-if**: "כמה תכניס דירת 2 חדרי שינה ב-De Pijp ל-4 אורחים?", "האם listing 28871 מתומחר נמוך?"
- **תשואה**: "אם אקנה ב-450,000 EUR עם עלויות 9,000 בשנה, מה התשואה?" (רק עם המספרים שלך)
- **רגולציה**: "כמה לילות מותר להשכיר באמסטרדם?", "איך הרגולציה משנה את הדירוג?", "האם משקיע יכול להשכיר דירה לתיירים?"
- **נתונים והקשר**: "אילו ערים יש בנתונים?", "מה המצב בברצלונה?"

**איך זה עובד:** נתב מעביר את השאלה לסוכן מתמחה (חוקר הזדמנויות, אנליסט מחירים, סוכן רגולציה או סוכן העשרה). הסוכן קורא לכלים קריאה-בלבד, וכל מספר בתשובה נבדק אוטומטית מול תוצאות הכלים ומצוטט [E#]. כל הזדמנות שמוצגת עוברת גם בדיקת רגולציה.

**מה הציון אומר:** הכנסה שנתית שמרנית מהשכרה (לילות מוזמנים × מחיר צפוי × ודאות). זו **לא** תשואה על השקעה ולא המלצת קנייה, והיא לא מגבילה את הלילות לתקרה החוקית: את זה עושה בדיקת הרגולציה."""
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
        for name in SPECIALISTS:
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
        for name in ["investigator", "price_analyst", "compliance"]:
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

        agent = INTENT_AGENT[intent]
        task = check.text if not city else f"{check.text}\n\n(City in context: {city})"
        try:
            res = run_agent(self.specs[agent], task, llm=self.llm, guard=self.guard, state=state,
                            config=self.config, history=history)
            answer, report = self.verify_and_revise(res.text, state)
            if agent != "compliance":
                answer, report = self.attach_regulation(answer, report, state, city)
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

    # ------------------------------------------------------------------ regulation, by construction
    def system_tool(self, name: str, args: dict, state: TurnState) -> tuple[dict, str]:
        """A tool call made by the system itself (not by a model): same evidence store and
        trace, no allow-list or budget, since no model chose it."""
        state.emit("tool_call", "regulation_check", tool=name, args=args)
        try:
            result = self.registry[name].handler(args)
        except Exception as e:                       # tools should not raise; be safe
            result = {"tool": name, "status": "tool_failure", "status_reason": f"{type(e).__name__}: {e}"}
        rec = state.store.register(name, "regulation_check", args, result)
        state.emit("tool_result", "regulation_check", tool=name, ref=rec.ref, status=result.get("status"),
                   evidence_id=result.get("evidence_id"))
        return result, rec.ref

    def _opportunities_in_turn(self, state: TurnState) -> tuple[str | None, list[dict], set[str]]:
        """The opportunities this turn presented (from rank_opportunities results) and the ones
        compliance_check already covered."""
        city, shown, checked, seen = None, [], set(), set()
        for r in state.store.records.values():
            res = r.result or {}
            if r.tool == "compliance_check" and res.get("status") == "ok":
                if res.get("action") == "ranking":
                    checked |= {f"rank:{x['rank']}" for x in res.get("results", [])}
                elif res.get("opportunity"):
                    checked.add(f"rank:{res['opportunity']['rank']}")
            if r.tool != "rank_opportunities" or res.get("status") != "ok":
                continue
            if res.get("action") not in ("top", "breakdown"):
                continue
            city = city or res.get("city")
            rows = res.get("results", [])[: 3 if res.get("action") == "top" else 1]
            for x in rows:
                if x.get("rank") and x["rank"] not in seen:
                    seen.add(x["rank"])
                    shown.append(x)
        return city, shown, checked

    def attach_regulation(self, answer: str, report: VerificationReport, state: TurnState,
                          city: str | None) -> tuple[str, VerificationReport]:
        """Every opportunity the answer presents gets the regulation check, whether or not the
        agent remembered to call it: a deterministic step after the agent, cited like the rest."""
        opp_city, shown, checked = self._opportunities_in_turn(state)
        todo = [x for x in shown if f"rank:{x['rank']}" not in checked]
        if not todo or self.capabilities.get("compliance_check"):
            return answer, report
        rules, rref = self.system_tool("regulation_rules", {"city": opp_city or city or ""}, state)
        if rules.get("status") != "ok":
            return answer, report
        state.emit("agent_start", "regulation_check", title=TITLES["regulation_check"], tools=["compliance_check"])
        rows = []
        for x in todo:
            res, ref = self.system_tool("compliance_check", {"action": "opportunity", "city": opp_city,
                                                             "rank": int(x["rank"])}, state)
            if res.get("status") == "ok":
                rows.append((res, ref))
        if not rows:
            return answer, report
        block = regulation_block(rows, rules, rref)
        final = f"{answer}\n\n{block}"
        report2 = verify(final, state.store, state.user_text)
        state.emit("verify", "verifier", title=TITLES["verifier"], after_regulation=True, **_short(report2))
        return final, report2

    # ------------------------------------------------------------------ deterministic fallback
    def fallback(self, text, intent, city, state, result, why):
        state.emit("guardrail", "fallback", guard="agent_unavailable", reason=why)
        allowed = {"rank_opportunities", "data_availability"}
        top, ref = self.guard.execute("fallback", allowed, "rank_opportunities",
                                      {"action": "top", "n": 10, **({"city": city} if city else {})}, state)
        lines = [f"> **מצב ללא סוכן** ({why}). מוצגת הטבלה הדטרמיניסטית, כפי שהאמנה מחייבת, "
                 "עם בדיקת הרגולציה.", ""]
        tables = {}
        if top.get("status") == "ok":
            cur = top.get("currency") or ""
            caps, cref = ({}, None)
            if not self.capabilities.get("compliance_check"):
                comp, cref = self.system_tool("compliance_check", {"action": "ranking", "city": top.get("city"),
                                                                   "n": 50}, state)
                caps = {x["rank"]: x for x in comp.get("results", [])} if comp.get("status") == "ok" else {}
            rref = None
            if caps:
                rules, rref = self.system_tool("regulation_rules", {"city": top.get("city")}, state)
                if intent == "regulation":
                    lines += regulation_summary_he(rules, rref)
            lines += [f"**10 ההזדמנויות המובילות ב-{top.get('city')}** (תווית: {top.get('label')}) [{ref}]"
                      + (f", ובתקרת הלילות החוקית [{cref}]" if caps else ""), ""]
            head = f"| דירוג | שכונה | סוג | חדרי שינה | ציון ({cur}/שנה) | מחיר ללילה | לילות | ודאות |"
            sep = "|---|---|---|---|---|---|---|---|"
            if caps:
                head += " לילות חוקיים | ציון בתקרה |"
                sep += "---|---|"
            lines += [head, sep]
            for r in top["results"]:
                line = (f"| {r['rank']} | {r['neighbourhood']} | {r['room_type']} | {r['bedrooms']} | "
                        f"{r['score_conservative_annual_revenue']:,.0f} | {r['expected_nightly_price']:,.0f} | "
                        f"{r['expected_booked_nights']:,.0f} | {r['price_certainty']:.2f} |")
                if caps:
                    c = caps.get(r["rank"]) or {}
                    line += (f" {c.get('legal_nights', 0):,.0f} | {c.get('score_within_cap', 0):,.0f} |"
                             if c else " - | - |")
                lines.append(line)
            lines += ["", "הציון הוא הכנסה שנתית שמרנית מהשכרה, לא תשואה על השקעה ולא המלצת קנייה."]
            if caps:
                lines += ["**רגולציה:** \"ציון בתקרה\" הוא אותה נוסחה עם הלילות שמותר להשכיר לפי מיקום כל נכס "
                          "(30 בשנה, 15 בשמונה רבעים). חדר פרטי הוא B&B: אין תקרת לילות, אבל נדרש היתר והמארח "
                          f"חייב לגור בדירה. משקיע שלא גר בנכס לא יכול להשכיר לתיירים כלל [{cref}, {rref}]."]
            tables["top"] = top["results"]
        else:
            avail, aref = self.guard.execute("fallback", allowed, "data_availability", {}, state)
            lines += [f"אין טבלת הזדמנויות זמינה: {top.get('status_reason')}.",
                      f"ערים בנתונים: {', '.join(c['city'] for c in avail.get('cities', [])) or 'אין'} [{aref}]"]
        lines += ["", "כדי לקבל חקירה מלאה עם סוכנים, הגדר `ANTHROPIC_API_KEY` (בסרגל הצד, בקובץ `.env` "
                  "או ב-Secrets של Streamlit)."]
        answer = "\n".join(lines)
        report = verify(answer, state.store, state.user_text)
        return result(answer, intent, "fallback", "fallback", city, report.as_dict(), why, tables)


REGIME_HE = {"holiday_rental": "דירה שלמה: השכרה לתיירים", "bed_and_breakfast": "חדר: B&B",
             "not_covered": "מחוץ לכללים"}


def regulation_block(rows: list[tuple[dict, str]], rules: dict, rules_ref: str) -> str:
    """The deterministic regulation section appended to an answer. Every number comes from a
    compliance_check result and carries its reference."""
    cur = rows[0][0].get("currency") or ""
    out = ["---", f"**⚖️ בדיקת רגולציה אוטומטית** ({rules.get('city')}, כללים שאומתו ב-{rules.get('verified_on')})", "",
           "השכרה לתיירים מותרת רק לתושב, בדירה שהיא מקום המגורים העיקרי שלו; משקיע שלא גר בנכס לא יכול "
           f"להשכיר אותו לטווח קצר [{rules_ref}].", "",
           f"| הזדמנות | משטר | לילות בציון | לילות מותרים | ציון בטבלה ({cur}) | ציון בתקרה ({cur}) | מקור |",
           "|---|---|---|---|---|---|---|"]
    notes = []
    for res, ref in rows:
        o, c = res["opportunity"], res["results"][0]
        w, nc = c.get("within_cap") or {}, c.get("night_cap") or {}
        legal = f"{w.get('legal_nights', 0):,.0f}" if c["regime"] == "holiday_rental" else "ללא תקרה"
        out.append(f"| #{o['rank']} {o['neighbourhood']}, {o['room_type']}, {o['bedrooms']} | {REGIME_HE[c['regime']]} | "
                   f"{w.get('booked_nights_in_score', 0):,.0f} | {legal} | {w.get('score_in_table', 0):,.0f} | "
                   f"{w.get('score_within_cap', 0):,.0f} | [{ref}] |")
        if c["regime"] == "holiday_rental" and nc.get("share_in_15_night_wijken") is not None:
            notes.append(f"#{o['rank']}: {100 * nc['share_in_15_night_wijken']:.0f}% מהנכסים ברבעים עם תקרת 15 לילות [{ref}]")
        elif c["regime"] == "bed_and_breakfast":
            notes.append(f"#{o['rank']}: B&B דורש היתר במכסה לרובע, והמארח גר בדירה [{ref}]")
    out += [""] + [f"- {n}" for n in notes]
    out.append("- הלילות בציון נמדדו ברובם לפני תקרת 15 הלילות (1.4.2026), ולכן הם מתארים את העבר ולא את "
               f"המותר היום [{rules_ref}]. זה לא ייעוץ משפטי.")
    return "\n".join(out)


def regulation_summary_he(rules: dict, ref: str) -> list[str]:
    if rules.get("status") != "ok":
        return []
    from agents.regulation import load_rules
    he = {r["id"]: r.get("rule_he") for r in load_rules(rules["city"])["rules"]}
    lines = [f"**כללי ההשכרה לטווח קצר ב-{rules['city']}** (אומתו ב-{rules['verified_on']}) [{ref}]", ""]
    for r in rules["results"]:
        src = rules["sources"].get(r["source"], {})
        lines.append(f"- **{r['id']}**: {he.get(r['id']) or r['rule']} "
                     f"({'מקור רשמי' if src.get('official') else 'מקור משני'}: {src.get('title', r['source'])})")
    return lines + [""]


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
    if re.search(r"regulat|licen[cs]e|permit|allowed|legal|b&b|night cap|tourist tax|רגולצי|רישיון|היתר|"
                 r"חוק|מותר|אסור|תקרת|ימים בשנה|לילות בשנה|מס תיירים|קנס", t):
        return {"intent": "regulation", "city": city, "reason": "keywords: regulation"}
    if re.search(r"listing|\bprice\b|nightly|per night|מחיר|ללילה|כמה (תכניס|יכניס|עולה)|roi|תשואה|what.?if", t):
        return {"intent": "price", "city": city, "reason": "keywords: price"}
    if re.search(r"data|cities|tax|נתונים|ערים|מס\b", t):
        return {"intent": "data", "city": city, "reason": "keywords: data"}
    return {"intent": "investigate", "city": city, "reason": "default: investigation"}
