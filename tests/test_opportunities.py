"""Tests for the opportunity score (src/models/opportunities.py) and the agent's tool
(src/models/opportunity_tool.py). No real data or saved models are needed.

Run from the repository root:   python -m pytest tests/test_opportunities.py -q
(without pytest:                python -m tests.test_opportunities)
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src.models.intervals import fit_cqr
from src.models.opportunities import (OpportunitySettings, assumption_sensitivity, bedroom_band,
                                      booked_nights, bootstrap_stability, cross_fitted_predictions,
                                      forward_check, listing_components, opportunity_table,
                                      save_opportunities, segment_key)
from src.models.opportunity_tool import OpportunityTool
from src.models.pipelines import make_model, make_preprocessor

S = OpportunitySettings(min_listings=20)
HOODS = ["De Pijp - Rivierenbuurt", "Centrum-West", "Noord-Oost"]


def listings(n=600, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "id": np.arange(1, n + 1),
        "snapshot_date": pd.Timestamp("2026-06-15"),
        "neighbourhood_cleansed": rng.choice(HOODS, n, p=[0.45, 0.45, 0.10]),
        "room_type": rng.choice(["Entire home/apt", "Private room"], n, p=[0.7, 0.3]),
        "bedrooms": rng.choice([0, 1, 2, 3, 4, np.nan], n, p=[.1, .3, .3, .2, .05, .05]),
        "accommodates": rng.integers(1, 7, n).astype(float),
        "latitude": rng.uniform(52.33, 52.40, n), "longitude": rng.uniform(4.85, 4.95, n),
        "minimum_nights": rng.choice([1, 2, 3, 5], n).astype(float),
        "number_of_reviews_ltm": rng.poisson(8, n).astype(float),
    })
    premium = d["neighbourhood_cleansed"].map(dict(zip(HOODS, [1.3, 1.5, 0.8])))
    d["price"] = (60 + 40 * d["accommodates"]) * premium * rng.lognormal(0, 0.2, n)
    return d


def components(d, seed=0):
    """Listing frame with stand-in out-of-fold values (no model needed)."""
    rng = np.random.default_rng(seed)
    pred = d["price"].to_numpy() * rng.lognormal(0, 0.1, len(d))
    oof = pd.DataFrame({"pred": pred, "lo": pred * 0.7, "hi": pred * 1.4}, index=d.index)
    return listing_components(d, oof, S)


# ------------------------------------------------------------------ the unit
def test_bedroom_bands_from_numbers_and_agent_strings():
    assert [bedroom_band(v) for v in [0, 1, 1.4, 2, 3, 7]] == ["0-1", "0-1", "0-1", "2", "3+", "3+"]
    assert bedroom_band(np.nan) == "unknown" and bedroom_band(None) == "unknown"
    assert [bedroom_band(v) for v in ["2", "3+", "3plus", "0-1", "studio", "4 bedrooms"]] == \
        ["2", "3+", "3+", "0-1", "0-1", "3+"]
    assert bedroom_band("lots") == "unknown"
    assert segment_key("De Pijp - Rivierenbuurt", "Entire home/apt", "3+") == \
        "de_pijp_rivierenbuurt__entire_home_apt__3plus"


# ------------------------------------------------------------------ component 2
def test_booked_nights_formula_cap_and_minimum_stay():
    d = pd.DataFrame({"number_of_reviews_ltm": [10, 0, 100, np.nan, 10],
                      "minimum_nights": [1, 2, 3, 2, 7]})
    n = booked_nights(d, S)
    assert n[0] == 10 / 0.5 * 3                       # 60: assumed 3-night stay
    assert n[1] == 0                                  # no recent reviews: 0 nights
    assert n[2] == 0.7 * 365                          # capped at 70% of the year
    assert np.isnan(n[3])                             # unknown stays unknown
    assert n[4] == 10 / 0.5 * 7                       # minimum stay longer than 3 nights
    assert booked_nights(d, S.with_(review_rate=0.25))[0] == 120


# ------------------------------------------------------------------ table and score
def test_score_is_the_product_of_the_three_components_and_ranks_follow_it():
    t = opportunity_table(components(listings()), S)
    ok = t[t["status"] == "ok"]
    assert np.allclose(ok["score"], ok["booked_nights"] * ok["expected_price"] * ok["price_certainty"])
    assert ok["rank"].tolist() == sorted(ok["rank"].tolist())
    assert (ok["score"].diff().dropna() <= 1e-9).all()        # rank 1 = highest score
    assert (ok["price_certainty"] <= 1).all() and (ok["price_certainty"] > 0).all()


def test_minimum_size_rule_and_unknown_bedrooms():
    lf = components(listings())
    t = opportunity_table(lf, S)
    small = t[t["n_listings"] < S.min_listings]
    assert len(small) and (small["status"] == "insufficient_evidence").all() and small["rank"].isna().all()
    assert (t[t["n_listings"] >= S.min_listings]["status"] == "ok").all()
    assert "unknown" not in set(t["bedroom_band"])
    assert t["n_listings"].sum() == (lf["bedroom_band"] != "unknown").sum()


# ------------------------------------------------------------------ out-of-fold components
def test_out_of_fold_prediction_never_uses_the_listings_own_price():
    d = listings(400)
    feats = ["accommodates", "latitude", "longitude", "neighbourhood_cleansed", "room_type"]
    prep = make_preprocessor(["accommodates", "latitude", "longitude"], [],
                             ["neighbourhood_cleansed", "room_type"])
    pipe = make_model("ridge", prep)
    q = lambda a: make_model("hist_gbm", prep).set_params(
        model=HistGradientBoostingRegressor(loss="quantile", quantile=a, max_iter=50, random_state=0))
    cqr = fit_cqr(q(0.1), q(0.9), d[feats].iloc[:300], d["price"].iloc[:300],
                  d[feats].iloc[300:], d["price"].iloc[300:])
    a = cross_fitted_predictions(pipe, cqr, d[feats], d["price"], d["id"], verbose=False)
    d2 = d.copy()
    d2.loc[0, "price"] = 50_000.0                      # change ONE listing's own price wildly
    b = cross_fitted_predictions(pipe, cqr, d2[feats], d2["price"], d2["id"], verbose=False)
    assert a.loc[0, "pred"] == b.loc[0, "pred"] and a.loc[0, "lo"] == b.loc[0, "lo"]
    assert a[["pred", "lo", "hi"]].notna().all().all() and (a["lo"] <= a["hi"]).all()


# ------------------------------------------------------------------ ranking checks
def test_ranking_checks_run_and_are_sensible():
    lf = components(listings())
    summary, per = bootstrap_stability(lf, S, n_boot=20, top_n=3, seed=1)
    assert 0 <= summary["mean top-3 overlap"] <= 1 and len(per) == summary["ranked segments"]
    sens = assumption_sensitivity(lf, S, review_rates=(0.3, 0.5), avg_stays=(3.0,), top_n=3)
    base = sens[sens["base"]].iloc[0]
    assert base["top-3 overlap with base"] == 1 and base["median score change (%)"] == 0
    later = listings(seed=5)
    fwd = forward_check(opportunity_table(lf, S), later, S, top_n=3)
    assert fwd["available"] and -1 <= fwd["rank correlation (score vs later revenue proxy)"] <= 1


# ------------------------------------------------------------------ the agent's tool
def _tool(tmp: Path) -> OpportunityTool:
    lf = components(listings())
    save_opportunities(opportunity_table(lf, S), lf,
                       {"city": "Amsterdam", "currency": "EUR", "label": "PROVISIONAL",
                        "snapshot_date": "2026-06-15", "model_version": "hist_gbm_amsterdam_test",
                        "settings": {**json.loads(json.dumps(S.__dict__, default=list))}},
                       tmp / "reports" / "amsterdam")
    return OpportunityTool.load(tmp, verbose=False)


def test_tool_top_breakdown_listings_and_fuzzy_names():
    with tempfile.TemporaryDirectory() as tmp:
        tool = _tool(Path(tmp))
        top = tool.handle_tool_call({"action": "top", "n": 3})
        assert top["status"] == "ok" and [r["rank"] for r in top["results"]] == [1, 2, 3]
        assert "not a return on investment" in top["guardrail"]
        r1 = top["results"][0]
        bd = tool.handle_tool_call({"action": "breakdown", "city": "amsterdam",
                                    "neighbourhood": r1["neighbourhood"].split(" - ")[0].lower(),
                                    "room_type": "entire" if r1["room_type"].startswith("Entire") else "private",
                                    "bedrooms": r1["bedrooms"]})
        assert bd["status"] == "ok" and bd["results"][0]["rank"] == 1
        assert len(bd["results"][0]["components"]) == 3 and "formula" in bd["results"][0]
        assert bd["evidence_id"].startswith("opp:amsterdam:")
        ls = tool.handle_tool_call({"action": "listings", "neighbourhood": r1["neighbourhood"],
                                    "room_type": r1["room_type"], "bedrooms": r1["bedrooms"], "n": 5})
        gaps = [x["gap_vs_expected_pct"] for x in ls["results"]]
        assert ls["status"] == "ok" and len(gaps) == 5 and gaps == sorted(gaps)
        filt = tool.handle_tool_call({"action": "top", "neighbourhood": "centrum", "bedrooms": 2})
        assert filt["status"] == "ok" and {r["neighbourhood"] for r in filt["results"]} == {"Centrum-West"}


def test_tool_refuses_without_numbers_and_never_raises():
    with tempfile.TemporaryDirectory() as tmp:
        tool = _tool(Path(tmp))
        small = tool.cities["Amsterdam"].table.query("status != 'ok'").iloc[0]
        r = tool.handle_tool_call({"action": "breakdown", "neighbourhood": small["neighbourhood_cleansed"],
                                   "room_type": small["room_type"], "bedrooms": small["bedroom_band"]})
        assert r["status"] == "insufficient_evidence"
        assert "score_conservative_annual_revenue" not in r["results"][0]
        assert tool.handle_tool_call({"action": "sell"})["status"] == "invalid_input"
        assert tool.handle_tool_call({"action": "top", "city": "Paris"})["status"] == "invalid_input"
        assert tool.handle_tool_call({"action": "breakdown", "neighbourhood": "Zuidas",
                                      "room_type": "entire", "bedrooms": 2})["status"] == "not_found"
        assert tool.handle_tool_call({"action": "breakdown", "neighbourhood": "e",
                                      "room_type": "entire", "bedrooms": 2})["status"] == "invalid_input"
        assert tool.handle_tool_call({"action": "top", "bedrooms": "many"})["status"] == "invalid_input"
        assert tool.handle_tool_call(None)["status"] == "invalid_input"
        keys = set(tool.handle_tool_call({"action": "top"}))
        assert keys == set(tool.handle_tool_call({"action": "sell"}))      # same keys every time


if __name__ == "__main__":                      # runner for machines without pytest
    import sys
    import traceback
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
