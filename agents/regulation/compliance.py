"""Deterministic compliance checks against the local rules (agents/regulation/*.json).

The opportunity score (src/models/opportunities.py) estimates what a segment EARNED, from
reviews of the twelve months before the snapshot. The rules decide what is ALLOWED now. This
module puts the two side by side, without changing the team's score:

* regime: a whole home is holiday rental (vakantieverhuur); a private or shared room is a
  bed & breakfast; a hotel room is outside these rules.
* who may rent: only a resident whose main residence it is (both regimes), so a non-resident
  investor may not rent to tourists at all.
* night cap per listing: 30 nights a year, 15 in the eight designated wijken (by the listing's
  coordinates, point in polygon), for holiday rental only; a B&B has no night cap.
* revenue within the cap: mean over the segment's listings of min(booked nights, its cap) x
  expected price x price certainty: the score's own formula with the nights a resident host
  may legally rent.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DIR = Path(__file__).resolve().parent
RULE_FILES = {"amsterdam": "amsterdam_rules.json"}
WIJK_FILES = {"amsterdam": "amsterdam_15_night_wijken.geojson"}

REGIMES = {
    "Entire home/apt": ("holiday_rental", "השכרת דירה שלמה לתיירים (vakantieverhuur)"),
    "Private room": ("bed_and_breakfast", "B&B: חדר בדירה שהמארח גר בה"),
    "Shared room": ("bed_and_breakfast", "B&B: חדר בדירה שהמארח גר בה"),
    "Hotel room": ("not_covered", "חדר מלון: מסחרי, מחוץ לכללי ההשכרה של דירות מגורים"),
}


def _slug(s) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(s).strip().lower()).strip("_")


@lru_cache(maxsize=None)
def load_rules(city: str) -> dict | None:
    """The rules file of a city, or None if the city has no curated rules."""
    name = RULE_FILES.get(_slug(city))
    if not name:
        return None
    return json.loads((DIR / name).read_text(encoding="utf-8"))


def rules_for_city(city: str, room_type: str | None = None, topic: str | None = None) -> list[dict]:
    reg = load_rules(city)
    if reg is None:
        return []
    out = []
    for r in reg["rules"]:
        if room_type and room_type not in r.get("applies_to", []):
            continue
        if topic and r.get("topic") != topic:
            continue
        out.append(r)
    return out


def regime_for(room_type: str) -> tuple[str, str]:
    return REGIMES.get(room_type, ("not_covered", "לא מכוסה בכללים"))


# ---------------------------------------------------------------------- night caps
@lru_cache(maxsize=None)
def _wijken(city: str) -> tuple:
    name = WIJK_FILES.get(_slug(city))
    if not name:
        return ()
    gj = json.loads((DIR / name).read_text(encoding="utf-8"))
    out = []
    for f in gj["features"]:
        rings = [np.asarray(poly[0], dtype=float) for poly in f["geometry"]["coordinates"]]
        out.append((f["properties"]["wijk"], int(f["properties"]["night_cap"]), tuple(rings)))
    return tuple(out)


def _inside(lon: np.ndarray, lat: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Vectorised ray casting: is each point inside the polygon ring (lon, lat)?"""
    x, y = ring[:, 0], ring[:, 1]
    inside = np.zeros(len(lon), dtype=bool)
    j = len(ring) - 1
    for i in range(len(ring)):
        cond = ((y[i] > lat) != (y[j] > lat)) & \
               (lon < (x[j] - x[i]) * (lat - y[i]) / np.where(y[j] == y[i], 1e-12, y[j] - y[i]) + x[i])
        inside ^= cond
        j = i
    return inside


def night_caps(city: str, lat, lon, neighbourhood) -> pd.DataFrame:
    """Per listing: the designated wijk it lies in (or '') and its holiday-rental night cap.

    A listing gets the 15-night cap only if it lies in a designated wijk AND in one of the
    neighbourhoods that contain those wijken, so a simplified boundary cannot leak a 15-night
    cap into a neighbouring district. Missing coordinates in those neighbourhoods give cap
    None (unknown); elsewhere the general cap applies."""
    lat = pd.to_numeric(pd.Series(lat), errors="coerce").to_numpy(dtype=float)
    lon = pd.to_numeric(pd.Series(lon), errors="coerce").to_numpy(dtype=float)
    hood = pd.Series(neighbourhood).astype(str).to_numpy()
    reg = load_rules(city)
    general = next((r["value"] for r in (reg or {}).get("rules", []) if r["id"] == "R2"), None)
    r3 = next((r for r in (reg or {}).get("rules", []) if r["id"] == "R3"), None)
    wijk = np.array([""] * len(lat), dtype=object)
    cap = np.array([general] * len(lat), dtype=object)
    if r3 is not None:
        ok = ~(np.isnan(lat) | np.isnan(lon))
        for name, c, rings in _wijken(city):
            hit = np.zeros(len(lat), dtype=bool)
            for ring in rings:
                hit[ok] |= _inside(lon[ok], lat[ok], ring)
            hit &= (wijk == "")
            wijk[hit] = name
        eligible = np.isin(hood, r3.get("inside_airbnb_neighbourhoods", []))
        cap[eligible & (wijk != "")] = r3["value"]
        cap[eligible & ~ok] = None
        wijk[~eligible] = ""
    return pd.DataFrame({"wijk_15_nights": wijk, "night_cap": cap})


# ---------------------------------------------------------------------- one opportunity
def segment_compliance(city: str, opp: dict, listings: pd.DataFrame | None) -> dict:
    """Compliance of one opportunity.

    opp: a row of the opportunity table (neighbourhood_cleansed, room_type, bedroom_band,
         booked_nights, expected_price, price_certainty, score, n_listings).
    listings: the segment's listings with at least booked_nights, latitude, longitude and
         neighbourhood_cleansed; optional accommodates and license. None = segment-level only.
    """
    reg = load_rules(city)
    room = opp["room_type"]
    regime, regime_he = regime_for(room)
    rules = rules_for_city(city, room)
    out = {"regime": regime, "regime_he": regime_he,
           "rule_ids": [r["id"] for r in rules],
           "who_may_rent": None, "investor_permitted": None, "night_cap": None,
           "within_cap": None, "flags": [], "conditions": []}
    if reg is None:
        out["status_reason"] = f"no curated regulation for {city}"
        return out
    booked = float(opp["booked_nights"])
    price, cert = float(opp["expected_price"]), float(opp["price_certainty"])

    if regime == "not_covered":
        out["flags"].append("hotel rooms are commercial use, outside the home-rental rules: check zoning")
        return out

    out["who_may_rent"] = "a resident whose main residence it is (registered there, 183+ nights a year)"
    out["investor_permitted"] = False
    out["conditions"] = [r["rule"] for r in rules if r["id"] in ("R1", "R5", "R6", "R7")]

    if regime == "holiday_rental":
        caps = None
        if listings is not None and len(listings) and {"latitude", "longitude"} <= set(listings.columns):
            caps = night_caps(city, listings["latitude"], listings["longitude"],
                              listings["neighbourhood_cleansed"])
            nights = pd.to_numeric(listings["booked_nights"], errors="coerce").to_numpy(dtype=float)
            cap_vals = caps["night_cap"].to_numpy(dtype=object)
            known = np.array([c is not None for c in cap_vals])
            conservative = np.array([15 if c is None else c for c in cap_vals], dtype=float)
            legal = np.nanmean(np.minimum(nights, conservative))
            n15 = int((caps["night_cap"] == 15).sum())
            out["night_cap"] = {
                "method": "per listing, by its coordinates",
                "listings": int(len(caps)),
                "listings_cap_15": n15,
                "listings_cap_30": int((caps["night_cap"] == 30).sum()),
                "listings_cap_unknown": int((~known).sum()),
                "share_in_15_night_wijken": round(n15 / len(caps), 3),
                "wijken": {k: int(v) for k, v in caps["wijk_15_nights"][caps["wijk_15_nights"] != ""]
                           .value_counts().items()},
                "share_listings_above_their_cap": round(float(np.nanmean(nights > conservative)), 3),
            }
        else:
            r3 = next(r for r in rules if r["id"] == "R3")
            maybe15 = opp["neighbourhood_cleansed"] in r3.get("inside_airbnb_neighbourhoods", [])
            cap = 15 if maybe15 else 30
            legal = min(booked, cap)
            out["night_cap"] = {"method": "segment level (no listing coordinates): "
                                + ("this neighbourhood contains 15-night wijken, so 15 is used as the "
                                   "conservative cap" if maybe15 else "general cap"),
                                "cap_used": cap}
        legal = float(legal)
        within = legal * price * cert
        out["within_cap"] = {
            "scenario": "a resident host renting their own home within the night cap",
            "booked_nights_in_score": round(booked, 1),
            "legal_nights": round(legal, 1),
            "score_within_cap": round(within, 0),
            "score_in_table": round(float(opp["score"]), 0),
            "reduction_pct": round(100 * (1 - within / float(opp["score"])), 1) if opp["score"] else None,
            "formula": f"{legal:.1f} legal nights x {price:,.2f} x {cert:.3f} = {within:,.0f} per year",
        }
        if booked > 30:
            out["flags"].append(f"the score assumes {booked:.0f} nights a year, above the legal cap; "
                                "observed nights above the cap describe past (possibly unlicensed) renting")
    else:   # bed & breakfast
        out["night_cap"] = {"method": "B&B has no night cap", "cap_used": None}
        out["within_cap"] = {
            "scenario": "a resident running a licensed B&B in their own home",
            "booked_nights_in_score": round(booked, 1), "legal_nights": round(booked, 1),
            "score_within_cap": round(float(opp["score"]), 0),
            "score_in_table": round(float(opp["score"]), 0), "reduction_pct": 0.0,
            "formula": "no night cap; the B&B permit, the host living there, max 40% of the floor "
                       "area (61 m2) and the wijk quota decide whether it is possible at all",
        }
        out["flags"].append("B&B permits are capped by a quota per wijk; check whether permits are available")

    if listings is not None and len(listings):
        if "accommodates" in listings.columns:
            acc = pd.to_numeric(listings["accommodates"], errors="coerce")
            share = float((acc > 4).mean()) if acc.notna().any() else None
            if share:
                out["share_listings_for_more_than_4_guests"] = round(share, 3)
                out["flags"].append(f"{share:.0%} of the listings take more than 4 guests, the legal maximum")
        if "license" in listings.columns:
            lic = listings["license"].astype("string").str.strip()
            missing = float((lic.isna() | (lic == "") | (lic.str.lower() == "nan")).mean())
            out["share_listings_without_registration_number"] = round(missing, 3)
            if missing > 0.1:
                out["flags"].append(f"{missing:.0%} of the listings show no registration number (required in every advert)")
    timing = next((r for r in reg["rules"] if r["id"] == "R10"), None)
    if timing:
        out["flags"].append(timing["rule"])
        out["rule_ids"] = sorted(set(out["rule_ids"]) | {"R10"}, key=lambda s: int(s[1:]))
    return out
