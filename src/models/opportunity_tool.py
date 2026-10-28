"""Read-only opportunity tool for the agent (README, "Opportunity definition and score").

The agent never reads the opportunity files itself. It calls one function with an `action`:

    top        the best-ranked opportunities (optionally filtered by neighbourhood, room type
               or bedrooms), e.g. "top 10 segments in Amsterdam"
    breakdown  one opportunity's three components and score, e.g. "score breakdown for
               De Pijp / entire home / 2 bedrooms"
    listings   the listings behind one opportunity (the evidence for an investigation),
               sorted from most below to most above the market

Every answer has the same keys:

    {"tool": "rank_opportunities", "action": "breakdown", "city": "Amsterdam",
     "status": "ok",                 # ok | insufficient_evidence | not_found | invalid_input
     "status_reason": null, "currency": "EUR",
     "label": "FINAL",               # FINAL | PROVISIONAL | SYNTHETIC (see opportunities.json)
     "results": [ {...}, ... ],
     "details": {...},               # action-specific extras (filters, counts), possibly empty
     "assumptions": {...},           # the occupancy assumptions and the score formula
     "guardrail": "...",             # what the numbers are not
     "table_version": "20261028_1012", "model_version": "...",
     "evidence_id": "opp:amsterdam:20261028_1012:breakdown:de_pijp_rivierenbuurt__entire_home_apt__2"}

Guardrails enforced in code: an opportunity with too few listings returns
`insufficient_evidence` without numbers; the score is labelled as conservative rental revenue
potential, never as a return on investment; the tool never raises at the agent.

Usage
-----
    from src.models.opportunity_tool import OpportunityTool, TOOL_SCHEMA
    tool = OpportunityTool.load(".")                     # from the repository root
    tool.handle_tool_call({"action": "top", "city": "Amsterdam", "n": 10})
    tool.handle_tool_call({"action": "breakdown", "city": "Amsterdam",
                           "neighbourhood": "De Pijp", "room_type": "entire home", "bedrooms": 2})

Command line (from the repository root):
    python -m src.models.opportunity_tool --city Amsterdam --top 10
    python -m src.models.opportunity_tool --city Amsterdam --breakdown "De Pijp" "entire home" 2
"""
from __future__ import annotations

import argparse
import difflib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.models.cities import city_slug
from src.models.opportunities import (UNKNOWN_BAND, OpportunitySettings, bedroom_band,
                                      load_opportunities, segment_key)

TOOL_NAME = "rank_opportunities"
MAX_N = 50
GUARDRAIL = ("The score is a conservative estimate of annual RENTAL REVENUE potential for a type "
             "of property in a neighbourhood (booked nights x expected nightly price x price "
             "certainty). It is not a return on investment, profit or property value: there are "
             "no purchase prices or costs in the data. Booked nights are an approximation from "
             "review counts with stated assumptions, not observed bookings. Rank 1 is a lead to "
             "investigate, not a buy recommendation.")

TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Ranks investment OPPORTUNITIES: combinations of a neighbourhood, a room type and a "
        "number of bedrooms (0-1, 2, 3+). Each has a score in local currency per year = expected "
        "booked nights x expected nightly price x price certainty (a conservative rental revenue "
        "estimate). Read-only. action='top' lists the best-ranked opportunities (optional filters); "
        "action='breakdown' explains one opportunity's components; action='listings' returns the "
        "existing listings behind one opportunity as evidence. If status is not 'ok', do not "
        "state any score or price: report status_reason. Never present the score as ROI, "
        "profit or a buy recommendation. Prices are in `currency`; never compare across "
        "currencies. Cite evidence_id for every claim."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["top", "breakdown", "listings"]},
            "city": {"type": "string",
                     "description": "City, e.g. 'Amsterdam'. Required when several cities are loaded."},
            "n": {"type": "integer", "description": f"Number of rows (default 10, max {MAX_N})."},
            "neighbourhood": {"type": "string",
                              "description": "Neighbourhood name or a unique part of it, e.g. 'De Pijp'."},
            "room_type": {"type": "string",
                          "description": "'entire home', 'private room', 'shared room' or 'hotel room'."},
            "bedrooms": {"description": "Bedrooms: a number, or a band '0-1', '2', '3+'."},
            "include_insufficient": {"type": "boolean",
                                     "description": "top only: also list opportunities with too few listings (without numbers)."},
        },
        "required": ["action"],
    },
}

ROOM_ALIASES = {"entire": "Entire home/apt", "whole": "Entire home/apt", "apartment": "Entire home/apt",
                "private": "Private room", "shared": "Shared room", "hotel": "Hotel room"}


def _clean(v, nd=2):
    if v is None or v is pd.NA:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), nd)
    return v


@dataclass
class CityTable:
    table: pd.DataFrame
    listings: pd.DataFrame
    meta: dict

    @property
    def city(self) -> str:
        return self.meta["city"]

    @property
    def currency(self) -> str:
        return self.meta.get("currency")

    @property
    def settings(self) -> OpportunitySettings:
        s = self.meta.get("settings", {})
        bands = tuple(tuple(b) for b in s.get("bedroom_bands", OpportunitySettings().bedroom_bands))
        return OpportunitySettings(**{**{k: v for k, v in s.items() if k != "bedroom_bands"},
                                      "bedroom_bands": bands})


@dataclass
class OpportunityTool:
    cities: dict = field(default_factory=dict)       # city -> CityTable
    evidence_log: Path | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, repo_root: Path | str = ".", evidence_log: Path | str | None = None,
             include_synthetic: bool = False, verbose: bool = True) -> "OpportunityTool":
        """Load every reports/<city>/opportunities.csv written by 06_experiments."""
        tables = {}
        for meta_path in sorted(Path(repo_root, "reports").glob("*/opportunities.json")):
            table, listings, meta = load_opportunities(meta_path.parent)
            if meta.get("label") == "SYNTHETIC" and not include_synthetic:
                continue
            tables[meta["city"]] = CityTable(table, listings, meta)
        if not tables:
            raise FileNotFoundError("no reports/<city>/opportunities.json found; run 06_experiments")
        tool = cls(tables, Path(evidence_log) if evidence_log else None)
        if verbose:
            for c, t in tables.items():
                print(f"{TOOL_NAME}: {c}: {t.meta['ranked_segments']} ranked opportunities "
                      f"({t.meta['segments']} segments), table {t.meta['table_version']}, {t.meta.get('label')}")
        return tool

    # ------------------------------------------------------------------ entry point
    def handle_tool_call(self, args: dict) -> dict:
        """What the agent framework calls. Never raises: errors come back as data."""
        try:
            args = dict(args or {})
            action = args.get("action")
            if action not in {"top", "breakdown", "listings"}:
                return self._answer(None, action, "invalid_input",
                                    "action must be 'top', 'breakdown' or 'listings'")
            ct, err = self._city(args.get("city"))
            if err:
                return self._answer(None, action, *err)
            if action == "top":
                out = self.top(ct, args)
            else:
                out = self.segment_action(ct, action, args)
        except Exception as e:                      # the agent must always get a record
            out = self._answer(None, args.get("action") if isinstance(args, dict) else None,
                               "invalid_input", f"could not process the call: {e}")
        if self.evidence_log is not None:
            with open(self.evidence_log, "a") as f:
                f.write(json.dumps({"request": args, "response": out}, default=str) + "\n")
        return out

    # ------------------------------------------------------------------ actions
    def top(self, ct: CityTable, args: dict) -> dict:
        n = self._n(args.get("n"), 10)
        t = ct.table
        filters = []
        if args.get("neighbourhood"):
            hood, err = self._match_neighbourhood(ct, args["neighbourhood"])
            if err:
                return self._answer(ct, "top", *err)
            t, filters = t[t["neighbourhood_cleansed"] == hood], filters + [hood]
        if args.get("room_type"):
            room, err = self._match_room(ct, args["room_type"])
            if err:
                return self._answer(ct, "top", *err)
            t, filters = t[t["room_type"] == room], filters + [room]
        if args.get("bedrooms") not in (None, ""):
            band = bedroom_band(args["bedrooms"], ct.settings)
            if band == UNKNOWN_BAND:
                return self._answer(ct, "top", "invalid_input",
                                    f"bedrooms {args['bedrooms']!r} not understood; use a number or "
                                    f"one of {[b[0] for b in ct.settings.bedroom_bands]}")
            t, filters = t[t["bedroom_band"] == band], filters + [f"{band} bedrooms"]
        ranked = t[t["status"] == "ok"].sort_values("rank").head(n)
        rows = [self._row(ct, r) for _, r in ranked.iterrows()]
        if args.get("include_insufficient"):
            rows += [self._row(ct, r) for _, r in t[t["status"] != "ok"].iterrows()]
        if not rows:
            return self._answer(ct, "top", "not_found",
                                f"no ranked opportunity matches {filters or 'the request'}")
        key = "top" + ("_" + "_".join(city_slug(f) for f in filters) if filters else "")
        return self._answer(ct, "top", "ok", None, rows, key=f"{key}_{n}",
                            extra={"ranked_opportunities_in_city": ct.meta["ranked_segments"],
                                   "filters": filters})

    def segment_action(self, ct: CityTable, action: str, args: dict) -> dict:
        missing = [k for k in ["neighbourhood", "room_type", "bedrooms"] if args.get(k) in (None, "")]
        if missing:
            return self._answer(ct, action, "invalid_input",
                                f"{action} needs neighbourhood, room_type and bedrooms; missing {missing}")
        hood, err = self._match_neighbourhood(ct, args["neighbourhood"])
        if err:
            return self._answer(ct, action, *err)
        room, err = self._match_room(ct, args["room_type"])
        if err:
            return self._answer(ct, action, *err)
        band = bedroom_band(args["bedrooms"], ct.settings)
        if band == UNKNOWN_BAND:
            return self._answer(ct, action, "invalid_input",
                                f"bedrooms {args['bedrooms']!r} not understood")
        key = segment_key(hood, room, band)
        seg = ct.table[ct.table["segment_key"] == key]
        if seg.empty:
            return self._answer(ct, action, "not_found",
                                f"no listings for {hood} / {room} / {band} bedrooms in the data", key=key)
        r = seg.iloc[0]
        if action == "breakdown":
            if r["status"] != "ok":
                return self._answer(ct, action, "insufficient_evidence",
                                    f"only {int(r['n_listings'])} priced listings for {hood} / {room} / "
                                    f"{band} bedrooms; at least {ct.settings.min_listings} are needed "
                                    "to score an opportunity", [self._row(ct, r)], key=key)
            return self._answer(ct, action, "ok", None, [self._row(ct, r, explain=True)], key=key)
        # listings: the evidence behind the opportunity
        n = self._n(args.get("n"), 20)
        L = ct.listings[ct.listings["segment_key"] == key].sort_values("gap_pct", na_position="last")
        rows = [{"listing_id": _clean(x["id"]), "asking_price": _clean(x["asking_price"]),
                 "expected_price": _clean(x["expected_price"]),
                 "interval_80": [_clean(x["price_low"]), _clean(x["price_high"])],
                 "position": x["position"] if isinstance(x["position"], str) else None,
                 "gap_vs_expected_pct": _clean(x["gap_pct"], 1),
                 "reviews_last_12m": _clean(x["reviews_last_12m"]),
                 "booked_nights": _clean(x["booked_nights"], 0),
                 "evidence_id": f"opp-listing:{city_slug(ct.city)}:{ct.meta['table_version']}:{_clean(x['id'])}"}
                for _, x in L.head(n).iterrows()]
        return self._answer(ct, action, "ok", None, rows, key=key,
                            extra={"opportunity_status": r["status"],
                                   "listings_in_opportunity": int(r["n_listings"]),
                                   "note": ("expected_price and interval_80 are out-of-fold: each listing "
                                            "was predicted by a model that did not see its price")})

    # ------------------------------------------------------------------ helpers
    def _city(self, city):
        if city is None:
            if len(self.cities) == 1:
                return next(iter(self.cities.values())), None
            return None, ("invalid_input", f"several cities loaded {sorted(self.cities)}: pass city")
        for c, t in self.cities.items():
            if city_slug(c) == city_slug(city):
                return t, None
        return None, ("invalid_input", f"no opportunity table for {city!r}; available: {sorted(self.cities)}")

    @staticmethod
    def _n(v, default):
        try:
            return max(1, min(MAX_N, int(v))) if v is not None else default
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _match_neighbourhood(ct: CityTable, query: str):
        names = sorted(ct.table["neighbourhood_cleansed"].unique())
        q = city_slug(query) if str(query).strip() else ""
        exact = [h for h in names if city_slug(h) == q]
        if exact:
            return exact[0], None
        part = [h for h in names if q and q in city_slug(h)]
        if len(part) == 1:
            return part[0], None
        if len(part) > 1:
            return None, ("invalid_input", f"{query!r} matches several neighbourhoods: {part}")
        close = difflib.get_close_matches(str(query), names, n=3, cutoff=0.5)
        return None, ("not_found", f"neighbourhood {query!r} not found" +
                      (f"; did you mean {close}?" if close else ""))

    @staticmethod
    def _match_room(ct: CityTable, query: str):
        rooms = sorted(ct.table["room_type"].unique())
        q = str(query).strip().lower()
        for r in rooms:
            if r.lower() == q:
                return r, None
        for alias, r in ROOM_ALIASES.items():
            if alias in q and r in rooms:
                return r, None
        part = [r for r in rooms if q and q in r.lower()]
        if len(part) == 1:
            return part[0], None
        return None, ("invalid_input", f"room type {query!r} not understood; use one of {rooms}")

    def _row(self, ct: CityTable, r, explain: bool = False) -> dict:
        ok = r["status"] == "ok"
        cur = ct.currency or ""
        row = {"rank": _clean(r["rank"]) if ok else None,
               "neighbourhood": r["neighbourhood_cleansed"], "room_type": r["room_type"],
               "bedrooms": r["bedroom_band"], "status": r["status"],
               "n_listings": int(r["n_listings"])}
        if ok:
            row.update({
                "score_conservative_annual_revenue": _clean(r["score"], 0),
                "expected_nightly_price": _clean(r["expected_price"]),
                "price_range_80": [_clean(r["price_low"]), _clean(r["price_high"])],
                "expected_booked_nights": _clean(r["booked_nights"], 0),
                "price_certainty": _clean(r["price_certainty"], 3),
                "expected_annual_revenue": _clean(r["expected_annual_revenue"], 0),
                "share_with_reviews_last_12m": _clean(r["share_active"], 3),
                "listings_priced_below_market": int(r["n_below_market"]),
                "median_asking_price": _clean(r["median_asking_price"]),
            })
            row["statement"] = (
                f"Rank {row['rank']} of {ct.meta['ranked_segments']}: {r['neighbourhood_cleansed']} / "
                f"{r['room_type']} / {r['bedroom_band']} bedrooms: about {cur} {r['score']:,.0f} per year "
                f"conservative rental revenue ({r['booked_nights']:.0f} nights x {cur} "
                f"{r['expected_price']:,.0f} x certainty {r['price_certainty']:.2f}), from "
                f"{int(r['n_listings'])} listings")
        if explain and ok:
            s = ct.settings
            row["components"] = [
                {"component": "1. expected nightly price", "value": _clean(r["expected_price"]),
                 "unit": f"{cur}/night",
                 "how": "median of the listings' out-of-fold model predictions (each listing "
                        "predicted by a model that did not see its price)"},
                {"component": "2. expected booked nights", "value": _clean(r["booked_nights"], 0),
                 "unit": "nights/year",
                 "how": (f"mean over the listings of reviews in the last 12 months / {s.review_rate:g} "
                         f"review rate x max({s.avg_stay_nights:g} nights, minimum stay), capped at "
                         f"{s.occupancy_cap:.0%} of the year; an approximation, not observed bookings")},
                {"component": "3. price certainty", "value": _clean(r["price_certainty"], 3),
                 "unit": "0-1",
                 "how": "median of (lower end of the 80% price range / expected price)"},
            ]
            row["formula"] = (f"{r['booked_nights']:.0f} x {r['expected_price']:,.2f} x "
                              f"{r['price_certainty']:.3f} = {r['score']:,.0f} {cur}/year")
        return row

    def _answer(self, ct, action, status, reason, results=None, key="na", extra=None) -> dict:
        meta = ct.meta if ct is not None else {}
        city = ct.city if ct is not None else None
        out = {"tool": TOOL_NAME, "action": action, "city": city, "status": status,
               "status_reason": reason, "currency": meta.get("currency"),
               "label": meta.get("label"), "results": results or [],
               "assumptions": (ct.settings.describe() if ct is not None else None),
               "guardrail": GUARDRAIL, "table_version": meta.get("table_version"),
               "model_version": meta.get("model_version"),
               "data_snapshot_date": meta.get("snapshot_date"),
               "details": extra or {},
               "evidence_id": (f"opp:{city_slug(city) if city else 'none'}:"
                               f"{meta.get('table_version', 'na')}:{action}:{key}")}
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Query the opportunity ranking (read-only).")
    ap.add_argument("--root", default=".")
    ap.add_argument("--city", default=None)
    ap.add_argument("--top", type=int, default=None, help="show the top N opportunities")
    ap.add_argument("--breakdown", nargs=3, metavar=("NEIGHBOURHOOD", "ROOM_TYPE", "BEDROOMS"))
    ap.add_argument("--listings", nargs=3, metavar=("NEIGHBOURHOOD", "ROOM_TYPE", "BEDROOMS"))
    a = ap.parse_args()
    tool = OpportunityTool.load(a.root)
    calls = []
    if a.top or not (a.breakdown or a.listings):
        calls.append({"action": "top", "city": a.city, "n": a.top or 10})
    for action, vals in [("breakdown", a.breakdown), ("listings", a.listings)]:
        if vals:
            calls.append({"action": action, "city": a.city, "neighbourhood": vals[0],
                          "room_type": vals[1], "bedrooms": vals[2]})
    for c in calls:
        print(json.dumps(tool.handle_tool_call(c), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
