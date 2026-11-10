"""Regulation tools: the local short-term-rental rules, and how they apply to an opportunity.

    regulation_rules   the curated rules of a city (agents/regulation/<city>_rules.json), each with
                       its official article, source link, effective date and verification date
    compliance_check   for one opportunity, or for the whole ranking: the legal regime, who may
                       rent, the night cap of each listing (15 or 30 by its location), the revenue
                       within the cap, and red flags (more than 4 guests, no registration number)

Both are deterministic and read-only. They never change the team's score: they put what the
rules allow next to it.
"""
from __future__ import annotations

import pandas as pd

from agents.regulation import load_rules, rules_for_city, segment_compliance
from agents.tools.base import ToolContext, ToolSpec, unavailable

DISCLAIMER = ("Curated summary of the official rules, verified on the date given; not legal advice. "
              "Rules change: check the official source before acting.")
TOPICS = ["who_may_rent", "night_cap", "escalation", "guests", "registration", "bed_and_breakfast",
          "fines", "tax", "data_timing"]


def _sources(reg: dict, rules: list[dict]) -> dict:
    keys = {r["source"] for r in rules}
    return {k: v for k, v in reg["sources"].items() if k in keys}


def regulation_rules_tool(ctx: ToolContext) -> ToolSpec:
    schema = {
        "name": "regulation_rules",
        "description": (
            "Read-only. The local short-term-rental rules of a city, curated from the official "
            "texts: who may rent to tourists, night caps (30 nights, 15 in designated wijken), "
            "maximum guests, registration and notice, bed & breakfast permits, escalation, fines "
            "and tourist tax. Each rule has an id (R1..), the article, the source (official or "
            "secondary), the date it took effect, and the date it was verified. Optional filters: "
            "room_type ('Entire home/apt', 'Private room', ...) and topic. Cite evidence_id and the "
            "rule id; say when a source is secondary. Cities without curated rules return "
            "not_found: then ask the Data Enrichment agent to search the web."),
        "input_schema": {"type": "object", "properties": {
            "city": {"type": "string"},
            "room_type": {"type": "string"},
            "topic": {"type": "string", "enum": TOPICS}},
            "required": ["city"]},
    }

    def handler(args: dict) -> dict:
        city = str(args["city"]).strip()
        reg = load_rules(city)
        base = {"tool": "regulation_rules", "city": city}
        if reg is None:
            return {**base, "status": "not_found", "results": [],
                    "status_reason": f"no curated regulation for {city!r}; available: Amsterdam. "
                                     "Use web search (Data Enrichment agent) and label it external.",
                    "evidence_id": f"rules:{city.lower()}:none"}
        room = args.get("room_type")
        if room:
            alias = {"entire": "Entire home/apt", "whole": "Entire home/apt", "apartment": "Entire home/apt",
                     "private": "Private room", "shared": "Shared room", "hotel": "Hotel room"}
            room = next((v for k, v in alias.items() if k in room.lower()), room)
        rules = rules_for_city(reg["city"], room, args.get("topic"))
        return {**base, "city": reg["city"], "status": "ok" if rules else "not_found",
                "status_reason": None if rules else "no rule matches these filters",
                "version": reg["version"], "verified_on": reg["verified_on"], "scope": reg["scope"],
                "results": [{k: r[k] for k in ("id", "topic", "title", "rule", "applies_to", "articles",
                                                "source", "effective_from", "value", "unit",
                                                "investor_impact") if k in r} for r in rules],
                "sources": _sources(reg, rules), "disclaimer": DISCLAIMER,
                "evidence_id": f"rules:{reg['city'].lower()}:{reg['version']}:"
                               f"{(room or 'all').replace('/', '_').replace(' ', '_').lower()}:{args.get('topic') or 'all'}"}

    return ToolSpec("regulation_rules", schema, handler)


def _segment_listings(ctx: ToolContext, ct, key: str) -> pd.DataFrame | None:
    """The segment's listings from the opportunity table, joined with their coordinates,
    capacity and registration from the processed listings."""
    seg = ct.listings[ct.listings["segment_key"] == key] if ct.listings is not None else None
    if seg is None or seg.empty:
        return None
    lst = ctx.listings()
    if lst is None:
        return seg
    cols = [c for c in ["id", "latitude", "longitude", "accommodates", "license"] if c in lst.columns]
    extra = lst[lst["location"] == ct.city][cols].drop_duplicates("id")
    seg = seg.copy()
    seg["id"] = pd.to_numeric(seg["id"], errors="coerce")
    extra = extra.assign(id=pd.to_numeric(extra["id"], errors="coerce"))
    return seg.merge(extra, on="id", how="left")


def _summary(r, comp: dict) -> dict:
    w = comp.get("within_cap") or {}
    nc = comp.get("night_cap") or {}
    return {"rank": int(r["rank"]), "neighbourhood": r["neighbourhood_cleansed"], "room_type": r["room_type"],
            "bedrooms": r["bedroom_band"], "regime": comp["regime"],
            "investor_permitted": comp["investor_permitted"],
            "score_in_table": w.get("score_in_table"), "booked_nights_in_score": w.get("booked_nights_in_score"),
            "legal_nights": w.get("legal_nights"), "score_within_cap": w.get("score_within_cap"),
            "reduction_pct": w.get("reduction_pct"),
            "share_in_15_night_wijken": nc.get("share_in_15_night_wijken")}


def compliance_check_tool(ctx: ToolContext) -> ToolSpec:
    schema = {
        "name": "compliance_check",
        "description": (
            "Read-only, deterministic. Applies the city's short-term-rental rules to the ranked "
            "opportunities. action='opportunity' (one opportunity: rank, or neighbourhood + "
            "room_type + bedrooms): the legal regime (whole home = holiday rental, room = B&B), "
            "who may rent (in Amsterdam only a resident in their main residence: "
            "investor_permitted=false), each listing's night cap by its location (30, or 15 in "
            "the eight designated wijken), legal nights, the score within the cap "
            "(legal nights x expected price x price certainty), and red flags (more than 4 guests, "
            "missing registration numbers, data older than the rule). action='ranking': the "
            "ranked opportunities re-ordered by the score within the cap (the resident-host "
            "scenario), with the original rank. Call it for every opportunity you present. "
            "Cite evidence_id and the rule ids."),
        "input_schema": {"type": "object", "properties": {
            "action": {"type": "string", "enum": ["opportunity", "ranking"]},
            "city": {"type": "string"},
            "rank": {"type": "integer", "minimum": 1},
            "neighbourhood": {"type": "string"},
            "room_type": {"type": "string"},
            "bedrooms": {"description": "a number, or '0-1', '2', '3+'"},
            "n": {"type": "integer", "minimum": 1, "maximum": 50}},
            "required": ["action", "city"]},
    }

    def handler(args: dict) -> dict:
        tool = ctx.opportunity_tool()
        if tool is None:
            return unavailable("compliance_check", "no opportunity table loaded: "
                               + ctx.errors.get("opportunities", "unknown"))
        ct, err = tool._city(args.get("city"))
        base = {"tool": "compliance_check", "action": args.get("action"), "city": args.get("city")}
        if err:
            return {**base, "status": err[0], "status_reason": err[1], "evidence_id": "reg:none"}
        reg = load_rules(ct.city)
        if reg is None:
            return {**base, "city": ct.city, "status": "not_found", "evidence_id": f"reg:{ct.city.lower()}:none",
                    "status_reason": f"no curated regulation for {ct.city}; ask the Data Enrichment agent"}
        base.update(city=ct.city, currency=ct.currency, rules_version=reg["version"],
                    label=ct.meta.get("label"), disclaimer=DISCLAIMER)
        ranked = ct.table[ct.table["status"] == "ok"].sort_values("rank")

        if args["action"] == "ranking":
            cache_key = ("compliance_ranking", ct.city, ct.meta.get("table_version"), reg["version"])
            if cache_key not in ctx._cache:            # deterministic: computed once per table
                rows = []
                for _, r in ranked.iterrows():
                    comp = segment_compliance(ct.city, r.to_dict(), _segment_listings(ctx, ct, r["segment_key"]))
                    rows.append(_summary(r, comp))
                rows = [x for x in rows if x["score_within_cap"] is not None]
                rows.sort(key=lambda x: -x["score_within_cap"])
                for i, x in enumerate(rows, 1):
                    x["rank_within_cap"] = i
                ctx._cache[cache_key] = rows
            rows = [dict(x) for x in ctx._cache[cache_key]]
            n = int(args.get("n") or 10)
            return {**base, "status": "ok", "status_reason": None,
                    "scenario": ("a resident host renting their own home within the night cap (whole "
                                 "homes) or running a licensed B&B (rooms). A non-resident investor may "
                                 "not rent to tourists at all."),
                    "results": rows[:n], "ranked_opportunities": len(rows),
                    "rule_ids": ["R1", "R2", "R3", "R7", "R10"],
                    "evidence_id": f"reg:{ct.city.lower()}:{reg['version']}:ranking:{ct.meta.get('table_version')}"}

        # one opportunity
        if args.get("rank"):
            hit = ranked[ranked["rank"] == int(args["rank"])]
            if hit.empty:
                return {**base, "status": "not_found", "evidence_id": "reg:none",
                        "status_reason": f"no ranked opportunity with rank {args['rank']}"}
        else:
            missing = [k for k in ("neighbourhood", "room_type", "bedrooms") if args.get(k) in (None, "")]
            if missing:
                return {**base, "status": "invalid_input", "evidence_id": "reg:none",
                        "status_reason": f"pass rank, or neighbourhood + room_type + bedrooms; missing {missing}"}
            from src.models.opportunities import bedroom_band
            hood, e1 = tool._match_neighbourhood(ct, args["neighbourhood"])
            room, e2 = tool._match_room(ct, args["room_type"])
            if e1 or e2:
                e = e1 or e2
                return {**base, "status": e[0], "status_reason": e[1], "evidence_id": "reg:none"}
            band = bedroom_band(args["bedrooms"])
            hit = ct.table[(ct.table["neighbourhood_cleansed"] == hood) & (ct.table["room_type"] == room)
                           & (ct.table["bedroom_band"] == band)]
            if hit.empty or hit.iloc[0]["status"] != "ok":
                return {**base, "status": "not_found", "evidence_id": "reg:none",
                        "status_reason": "this opportunity is not ranked (too few listings or not in the table)"}
        r = hit.iloc[0]
        comp = segment_compliance(ct.city, r.to_dict(), _segment_listings(ctx, ct, r["segment_key"]))
        return {**base, "status": "ok", "status_reason": None,
                "opportunity": {"rank": int(r["rank"]), "neighbourhood": r["neighbourhood_cleansed"],
                                "room_type": r["room_type"], "bedrooms": r["bedroom_band"],
                                "n_listings": int(r["n_listings"])},
                "results": [comp],
                "evidence_id": f"reg:{ct.city.lower()}:{reg['version']}:opp:{r['segment_key']}"}

    return ToolSpec("compliance_check", schema, handler)
