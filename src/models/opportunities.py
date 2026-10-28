"""Opportunity score: what the system ranks, and how (README, "Opportunity definition and score").

An **opportunity** is a combination of a neighbourhood, a room type and a number of bedrooms
(a *segment*). Existing listings are evidence about each opportunity; they are not the things
ranked, since they are not for sale. Each opportunity's score is built from three components:

1. **Expected nightly price** (currency/night): the median of the segment's listings'
   *out-of-fold* predictions. Each listing is predicted by a model that did not see it, so a
   listing's own price never sets its own estimate (`cross_fitted_predictions`).
2. **Expected booked nights** (nights/year): a stated approximation from the review rate,
   the method Inside Airbnb uses for `estimated_occupancy_l365d`:
   reviews in the last 12 months / review rate x average stay, where the average stay is the
   assumed stay or the listing's minimum stay if longer, capped at a share of the year
   (`booked_nights`). Listings without reviews in 12 months count as 0 nights (conservative),
   and the opportunity's value is the **mean** over its listings: the expected booked nights of
   a property of this type (a median would turn an opportunity with many inactive listings
   into an exact 0).
3. **Price certainty** (0-1): lower end of the listing's 80% conformal interval / its
   expected price, median over the segment. 1 = precise; lower = more spread.

    score (currency/year) = booked nights x expected nightly price x price certainty

which is approximately the booked nights times the lower end of the 80% price range: a
**conservative annual rental revenue**. It is revenue potential, not return on investment:
the data has no purchase prices or operating costs. Segments with fewer than
`min_listings` priced listings are reported as `insufficient_evidence` and not ranked.

The ranking checks used by notebook 06 also live here: bootstrap stability, sensitivity to
the occupancy assumptions, agreement of component 2 with Inside Airbnb's own estimate,
segment-level price accuracy, and a forward check against a later snapshot.
"""
from __future__ import annotations

import json
import math
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

from src.models.cities import city_slug
from src.models.data import ID_COL, SNAPSHOT_COL, TARGET

SEGMENT_COLS = ["neighbourhood_cleansed", "room_type", "bedroom_band"]
UNKNOWN_BAND = "unknown"
INSIDE_AIRBNB_OCCUPANCY = "estimated_occupancy_l365d"


@dataclass(frozen=True)
class OpportunitySettings:
    """Every assumption of the score, in one place (README: "stated, not estimated")."""
    min_listings: int = 20               # priced listings needed to rank a segment
    review_rate: float = 0.5             # share of stays that leave a review
    avg_stay_nights: float = 3.0         # assumed stay; the listing's minimum stay if longer
    occupancy_cap: float = 0.7           # booked nights capped at this share of the year
    # (label, highest bedroom count in the band; None = no upper limit)
    bedroom_bands: tuple = (("0-1", 1), ("2", 2), ("3+", None))
    reviews_col: str = "number_of_reviews_ltm"
    min_nights_col: str = "minimum_nights"

    def with_(self, **changes) -> "OpportunitySettings":
        return OpportunitySettings(**{**asdict(self), **changes})

    def describe(self) -> dict:
        return {"unit": "neighbourhood x room type x bedrooms",
                "bedroom_bands": [b[0] for b in self.bedroom_bands],
                "min_listings": self.min_listings,
                "occupancy": (f"reviews in last 12 months / {self.review_rate:g} review rate x "
                              f"max({self.avg_stay_nights:g} nights, minimum stay), capped at "
                              f"{self.occupancy_cap:.0%} of the year; no recent reviews = 0 nights"),
                "score": "booked nights x expected nightly price x price certainty (per year)",
                "review_rate": self.review_rate, "avg_stay_nights": self.avg_stay_nights,
                "occupancy_cap": self.occupancy_cap}


# ---------------------------------------------------------------------- unit
def bedroom_band(bedrooms, settings: OpportunitySettings = OpportunitySettings()):
    """Bedroom count(s) -> band label ('0-1', '2', '3+'); missing -> 'unknown'.

    Accepts a Series, a number, or a string such as '2', '3+', 'studio' (the agent's input).
    """
    if isinstance(bedrooms, pd.Series):
        return bedrooms.map(lambda v: bedroom_band(v, settings))
    if bedrooms is None:
        return UNKNOWN_BAND
    if isinstance(bedrooms, str):
        t = bedrooms.strip().lower().replace(" ", "")
        labels = {b[0].lower(): b[0] for b in settings.bedroom_bands}
        if t in labels:
            return labels[t]
        if t in {"studio", "0"}:
            return bedroom_band(0, settings)
        t = t.replace("plus", "+").rstrip("+").replace("bedrooms", "").replace("bedroom", "")
        try:
            return bedroom_band(float(t), settings)
        except ValueError:
            return UNKNOWN_BAND
    try:
        v = float(bedrooms)
    except (TypeError, ValueError):
        return UNKNOWN_BAND
    if math.isnan(v) or v < 0:
        return UNKNOWN_BAND
    v = int(round(v))
    for label, upper in settings.bedroom_bands:
        if upper is None or v <= upper:
            return label
    return UNKNOWN_BAND


def _slug(text) -> str:
    return city_slug(str(text).replace("+", "plus")) if str(text).strip() else "na"


def segment_key(neighbourhood, room_type, band) -> str:
    """Stable id of a segment, e.g. 'de_pijp_rivierenbuurt__entire_home_apt__2'."""
    return "__".join(_slug(x) for x in [neighbourhood, room_type, band])


# ---------------------------------------------------------------------- component 2
def booked_nights(df: pd.DataFrame, settings: OpportunitySettings = OpportunitySettings()
                  ) -> pd.Series:
    """Booked nights per year from the review rate (Inside Airbnb's occupancy model).

    nights = reviews in the last 12 months / review_rate x max(avg_stay, minimum_nights),
    capped at occupancy_cap x 365. Missing review counts give NaN (excluded from medians).
    """
    reviews = pd.to_numeric(df[settings.reviews_col], errors="coerce").astype(float)
    if settings.min_nights_col in df.columns:
        min_nights = pd.to_numeric(df[settings.min_nights_col], errors="coerce").astype(float)
        stay = np.fmax(settings.avg_stay_nights, min_nights.fillna(settings.avg_stay_nights))
    else:
        stay = pd.Series(settings.avg_stay_nights, index=df.index)
    nights = reviews / settings.review_rate * stay
    return nights.clip(upper=settings.occupancy_cap * 365).rename("booked_nights")


# ---------------------------------------------------------------------- components 1 and 3
def assign_folds(ids: pd.Series, n_folds: int = 5, seed: int = 42) -> np.ndarray:
    """Fold number per row; all rows of one listing in the same fold (grouped, seeded)."""
    uniq = pd.unique(ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = dict(zip(uniq[order], np.arange(len(uniq)) % n_folds))
    return ids.map(fold_of).to_numpy()


def cross_fitted_predictions(pipe, cqr: dict, X: pd.DataFrame, y: pd.Series, ids: pd.Series,
                             n_folds: int = 5, seed: int = 42, log_target: bool = True,
                             verbose: bool = True) -> pd.DataFrame:
    """Out-of-fold expected price and 80% interval for every row.

    In each fold, an unfitted copy of the selected model (same settings, same kNN k) and of
    the two quantile models is trained on the other folds and predicts the held-out fold, so
    no listing is ever predicted by a model that saw its price. The conformal widening Q comes
    from the calibration of 04_modeling (an approximation: the fold models are trained on a
    similar number of listings); `oof_coverage` checks it.
    """
    to_m = np.log1p if log_target else (lambda v: np.asarray(v, float))
    from_m = np.expm1 if log_target else (lambda v: np.asarray(v, float))
    folds = assign_folds(ids, n_folds, seed)
    out = pd.DataFrame(index=X.index, columns=["pred", "lo", "hi"], dtype=float)
    z = np.log1p(np.asarray(y, float))
    for k in range(n_folds):
        tr, va = folds != k, folds == k
        if not va.any():
            continue
        point = clone(pipe).fit(X[tr], to_m(np.asarray(y, float)[tr]))
        lo_m = clone(cqr["lo"]).fit(X[tr], z[tr])
        hi_m = clone(cqr["hi"]).fit(X[tr], z[tr])
        lo = np.expm1(lo_m.predict(X[va]) - cqr["q"])
        hi = np.expm1(hi_m.predict(X[va]) + cqr["q"])
        out.loc[va, "pred"] = from_m(point.predict(X[va]))
        out.loc[va, "lo"] = np.minimum(lo, hi)
        out.loc[va, "hi"] = np.maximum(lo, hi)
        if verbose:
            print(f"fold {k + 1}/{n_folds}: {int(va.sum()):,} listings predicted by models "
                  f"trained on the other {int(tr.sum()):,}")
    out["fold"] = folds
    return out


def oof_coverage(listing_frame: pd.DataFrame) -> dict:
    f = listing_frame.dropna(subset=["asking_price", "price_low", "price_high"])
    inside = (f["asking_price"] >= f["price_low"]) & (f["asking_price"] <= f["price_high"])
    return {"listings": len(f), "coverage": float(inside.mean()),
            "median_width": float((f["price_high"] - f["price_low"]).median())}


# ---------------------------------------------------------------------- listing-level evidence
def listing_components(listings: pd.DataFrame, oof: pd.DataFrame,
                       settings: OpportunitySettings = OpportunitySettings()) -> pd.DataFrame:
    """One row per listing: its segment, asking price, out-of-fold components and flag."""
    f = pd.DataFrame({
        ID_COL: listings[ID_COL].to_numpy(),
        "neighbourhood_cleansed": listings["neighbourhood_cleansed"].astype(str).to_numpy(),
        "room_type": listings["room_type"].astype(str).to_numpy(),
        "bedrooms": pd.to_numeric(listings["bedrooms"], errors="coerce").to_numpy(),
        "asking_price": pd.to_numeric(listings[TARGET], errors="coerce").to_numpy(),
        "expected_price": oof["pred"].to_numpy(),
        "price_low": oof["lo"].to_numpy(),
        "price_high": oof["hi"].to_numpy(),
        "reviews_last_12m": pd.to_numeric(listings[settings.reviews_col], errors="coerce").to_numpy(),
        "minimum_nights": (pd.to_numeric(listings[settings.min_nights_col], errors="coerce").to_numpy()
                           if settings.min_nights_col in listings.columns else np.nan),
    })
    if SNAPSHOT_COL in listings.columns:
        f[SNAPSHOT_COL] = pd.to_datetime(listings[SNAPSHOT_COL]).dt.strftime("%Y-%m-%d").to_numpy()
    if INSIDE_AIRBNB_OCCUPANCY in listings.columns:
        f["inside_airbnb_occupancy"] = pd.to_numeric(listings[INSIDE_AIRBNB_OCCUPANCY],
                                                     errors="coerce").to_numpy()
    f["bedroom_band"] = bedroom_band(f["bedrooms"], settings)
    f["price_certainty"] = (f["price_low"] / f["expected_price"]).clip(0, 1)
    f["position"] = np.select([f["asking_price"] < f["price_low"], f["asking_price"] > f["price_high"]],
                              ["below_interval", "above_interval"], "within_interval")
    f.loc[f["asking_price"].isna(), "position"] = None
    f["gap_pct"] = 100 * (f["asking_price"] / f["expected_price"] - 1)
    f["segment_key"] = [segment_key(*r) for r in f[SEGMENT_COLS].itertuples(index=False)]
    return with_occupancy(f, settings)


def with_occupancy(listing_frame: pd.DataFrame, settings: OpportunitySettings) -> pd.DataFrame:
    """(Re)compute component 2 for each listing under `settings` (used by the sensitivity check)."""
    f = listing_frame.copy()
    f["booked_nights"] = booked_nights(
        f.rename(columns={"reviews_last_12m": settings.reviews_col,
                          "minimum_nights": settings.min_nights_col}), settings).to_numpy()
    return f


# ---------------------------------------------------------------------- the opportunity table
def opportunity_table(listing_frame: pd.DataFrame,
                      settings: OpportunitySettings = OpportunitySettings()) -> pd.DataFrame:
    """One row per segment: the three components, the score, the status and the rank."""
    f = listing_frame[listing_frame["bedroom_band"] != UNKNOWN_BAND]
    g = f.groupby(SEGMENT_COLS, observed=True, sort=True)
    t = pd.DataFrame({
        "segment_key": g["segment_key"].first(),
        "n_listings": g.size(),
        "expected_price": g["expected_price"].median(),
        "price_low": g["price_low"].median(),
        "price_high": g["price_high"].median(),
        "price_certainty": g["price_certainty"].median(),
        "booked_nights": g["booked_nights"].mean(),      # expected value: zeros (inactive) count
        "share_active": g["reviews_last_12m"].apply(lambda s: float((s > 0).mean())),
        "median_asking_price": g["asking_price"].median(),
        "n_below_market": g["position"].apply(lambda s: int((s == "below_interval").sum())),
    })
    if "inside_airbnb_occupancy" in f.columns:
        t["inside_airbnb_occupancy"] = g["inside_airbnb_occupancy"].median()
    t["expected_annual_revenue"] = t["booked_nights"] * t["expected_price"]
    t["score"] = t["booked_nights"] * t["expected_price"] * t["price_certainty"]
    t["status"] = np.where(t["n_listings"] >= settings.min_listings, "ok", "insufficient_evidence")
    ok = t["status"] == "ok"
    t["rank"] = pd.Series(np.nan, index=t.index)
    t.loc[ok, "rank"] = t.loc[ok, "score"].rank(ascending=False, method="min")
    t = t.reset_index().sort_values(["rank", "n_listings"], ascending=[True, False],
                                    na_position="last").reset_index(drop=True)
    t["rank"] = t["rank"].astype("Int64")
    return t


def unknown_band_count(listing_frame: pd.DataFrame) -> int:
    return int((listing_frame["bedroom_band"] == UNKNOWN_BAND).sum())


# ---------------------------------------------------------------------- ranking checks
def _top(table: pd.DataFrame, n: int) -> list[str]:
    ok = table[table["status"] == "ok"].sort_values("rank")
    return ok["segment_key"].head(n).tolist()


def _spearman(x: pd.Series, y: pd.Series) -> float:
    """Rank correlation; NaN (no warning) when either side is constant or too short."""
    x, y = pd.Series(x).reset_index(drop=True), pd.Series(y).reset_index(drop=True)
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x[ok].nunique() < 2 or y[ok].nunique() < 2:
        return np.nan
    return float(x[ok].rank().corr(y[ok].rank()))


def _rank_corr(a: pd.DataFrame, b: pd.DataFrame) -> float:
    m = (a[a["status"] == "ok"][["segment_key", "score"]]
         .merge(b[b["status"] == "ok"][["segment_key", "score"]], on="segment_key"))
    return _spearman(m["score_x"], m["score_y"])


def bootstrap_stability(listing_frame: pd.DataFrame, settings: OpportunitySettings,
                        n_boot: int = 200, top_n: int = 10, seed: int = 0
                        ) -> tuple[dict, pd.DataFrame]:
    """How much does the ranking depend on which listings happen to be in the data?

    Listings are resampled with replacement (one row per listing, so this is a cluster
    bootstrap) and the table is rebuilt from the same per-listing components; the models are
    not refitted, so this measures sampling noise in the aggregation, not model uncertainty.
    Returns a summary and, per ranked segment, how often it lands in the top `top_n` and the
    5th-95th percentile of its rank.
    """
    base = opportunity_table(listing_frame, settings)
    base_top = set(_top(base, top_n))
    rng = np.random.default_rng(seed)
    n = len(listing_frame)
    ranks, overlaps, corrs = [], [], []
    for _ in range(n_boot):
        boot = listing_frame.iloc[rng.integers(0, n, n)]
        t = opportunity_table(boot, settings)
        ranks.append(t.set_index("segment_key")["rank"])
        overlaps.append(len(base_top & set(_top(t, top_n))) / max(1, len(base_top)))
        corrs.append(_rank_corr(base, t))
    R = pd.concat(ranks, axis=1).astype(float)
    ok = base[base["status"] == "ok"].set_index("segment_key")
    per = pd.DataFrame({"rank": ok["rank"].astype(int)})
    Rk = R.reindex(per.index)
    per[f"P(top {top_n})"] = (Rk <= top_n).mean(axis=1)
    per["rank p05"] = Rk.quantile(0.05, axis=1)
    per["rank p95"] = Rk.quantile(0.95, axis=1)
    per["P(ranked)"] = Rk.notna().mean(axis=1)
    summary = {"n_boot": n_boot, "top_n": top_n,
               f"mean top-{top_n} overlap": float(np.mean(overlaps)),
               "median rank correlation": float(np.nanmedian(corrs)),
               "ranked segments": int(len(per))}
    return summary, per.sort_values("rank")


def assumption_sensitivity(listing_frame: pd.DataFrame, settings: OpportunitySettings,
                           review_rates=(0.3, 0.5, 0.7), avg_stays=(2.0, 3.0, 4.0),
                           caps=(0.7,), top_n: int = 10) -> pd.DataFrame:
    """Rebuild the ranking under alternative occupancy assumptions and compare with the base.

    A review-rate change scales every listing's nights by the same factor, so it moves the
    score's level but changes the order only where the cap binds; the average stay interacts
    with minimum stays and can reorder segments.
    """
    base = opportunity_table(listing_frame, settings)
    base_top = set(_top(base, top_n))
    base_scores = base.set_index("segment_key")["score"]
    rows = []
    for rr in review_rates:
        for st in avg_stays:
            for cap in caps:
                s = settings.with_(review_rate=rr, avg_stay_nights=st, occupancy_cap=cap)
                t = opportunity_table(with_occupancy(listing_frame, s), s)
                common = t.set_index("segment_key")["score"].reindex(base_scores.index)
                rows.append({"review_rate": rr, "avg_stay_nights": st, "occupancy_cap": cap,
                             "base": (rr, st, cap) == (settings.review_rate,
                                                       settings.avg_stay_nights,
                                                       settings.occupancy_cap),
                             f"top-{top_n} overlap with base": len(base_top & set(_top(t, top_n))) / max(1, len(base_top)),
                             "rank correlation with base": _rank_corr(base, t),
                             "median score change (%)": float(np.nanmedian(100 * (common / base_scores - 1)))})
    return pd.DataFrame(rows)


def occupancy_agreement(listing_frame: pd.DataFrame, table: pd.DataFrame | None = None) -> dict:
    """Component 2 vs Inside Airbnb's own `estimated_occupancy_l365d`, when the column exists.

    Our default assumptions follow Inside Airbnb's published method, so agreement shows the
    approximation is implemented consistently; it does not show that it equals real bookings.
    """
    if "inside_airbnb_occupancy" not in listing_frame.columns:
        return {"available": False}
    f = listing_frame.dropna(subset=["booked_nights", "inside_airbnb_occupancy"])
    out = {"available": True, "listings": len(f),
           "rank correlation (listings)": _spearman(f["booked_nights"], f["inside_airbnb_occupancy"]),
           "median ours": float(f["booked_nights"].median()),
           "median Inside Airbnb": float(f["inside_airbnb_occupancy"].median())}
    if table is not None and "inside_airbnb_occupancy" in table.columns:
        ok = table[table["status"] == "ok"]
        out["rank correlation (segments)"] = _spearman(ok["booked_nights"], ok["inside_airbnb_occupancy"])
    return out


def segment_price_check(table: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Component 1 at segment level: expected price vs the segment's median asking price."""
    ok = table[table["status"] == "ok"].copy()
    ok["abs error"] = (ok["expected_price"] - ok["median_asking_price"]).abs()
    ok["abs error (%)"] = 100 * ok["abs error"] / ok["median_asking_price"]
    summary = {"ranked segments": len(ok), "MAE of segment medians": float(ok["abs error"].mean()),
               "median abs error (%)": float(ok["abs error (%)"].median())}
    cols = ["neighbourhood_cleansed", "room_type", "bedroom_band", "n_listings", "expected_price",
            "median_asking_price", "abs error", "abs error (%)"]
    return summary, ok[cols].sort_values("abs error (%)", ascending=False)


def forward_check(table_early: pd.DataFrame, later_listings: pd.DataFrame,
                  settings: OpportunitySettings, top_n: int = 10) -> dict:
    """Did the opportunities ranked high at an earlier snapshot earn more at a later one?

    `later_listings` is the later snapshot (prepared listings with asking price, review count
    and bedrooms). The later outcome per segment is its observed revenue proxy: median asking
    price x mean booked nights, both from the later snapshot.
    """
    f = later_listings.copy()
    f["bedroom_band"] = bedroom_band(pd.to_numeric(f["bedrooms"], errors="coerce"), settings)
    f["booked_nights"] = booked_nights(f, settings)
    f[TARGET] = pd.to_numeric(f[TARGET], errors="coerce")
    g = f.groupby(SEGMENT_COLS, observed=True)
    later = pd.DataFrame({"later_price": g[TARGET].median(), "later_nights": g["booked_nights"].mean(),
                          "later_listings": g.size()}).reset_index()
    later["later_revenue_proxy"] = later["later_price"] * later["later_nights"]
    m = table_early[table_early["status"] == "ok"].merge(later, on=SEGMENT_COLS, how="inner")
    if len(m) < 3:
        return {"available": False, "reason": "too few segments in both snapshots"}
    top = m["rank"] <= top_n
    return {"available": True, "segments in both snapshots": len(m),
            "rank correlation (score vs later revenue proxy)": _spearman(m["score"], m["later_revenue_proxy"]),
            f"median later revenue proxy, top {top_n}": float(m.loc[top, "later_revenue_proxy"].median()),
            "median later revenue proxy, the rest": float(m.loc[~top, "later_revenue_proxy"].median())}


# ---------------------------------------------------------------------- model and partitions
def confirmatory_run_done(repo_root: Path | str, city: str, split_version: str) -> bool:
    """True if 05_model_evaluation has made its one confirmatory test run for this city and split."""
    log = Path(repo_root) / "evaluation" / "test_runs.jsonl"
    if not log.exists():
        return False
    for line in log.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (r.get("city") == city and r.get("split_version") == split_version
                and r.get("label") == "CONFIRMATORY" and not r.get("synthetic", False)):
            return True
    return False


def load_model_bundle(repo_root: Path | str, city: str, synthetic: bool = False,
                      model: str | None = None) -> tuple[object, dict, dict, str]:
    """(pipeline, cqr, metadata, how it was chosen) for one city.

    The model is the one 05_model_evaluation selected (reports/<city>/model_card_metrics.json);
    before that, 04_modeling's development pick, with a warning (the same rule as the
    prediction tool). The artifact's hash is checked against the metadata.
    """
    from src.models.intervals import load_cqr
    from src.models.predict import _selected_model, _sha256_file

    models_dir = Path(repo_root) / "models"
    metas = [json.loads(p.read_text()) | {"_path": str(p)} for p in models_dir.glob("*_*.json")]
    metas = [m for m in metas if "model" in m and "split_version" in m and m.get("city") == city
             and bool(m.get("synthetic_data", False)) == synthetic]
    if not metas:
        raise FileNotFoundError(f"no {'synthetic' if synthetic else 'real-data'} model for "
                                f"{city!r} in {models_dir}; run 04_modeling first")
    how = "requested"
    if model is None:
        model = None if synthetic else _selected_model(Path(repo_root), city)
        how = "selected by 05_model_evaluation"
        if model is None:
            model = max(metas, key=lambda m: m["version"]).get("development_pick")
            how = "04_modeling development pick (05_model_evaluation has not selected a model yet)"
            if not synthetic:
                warnings.warn(f"{city}: {how}; the ranking is provisional.")
    cands = [m for m in metas if m["model"] == model]
    if not cands:
        raise FileNotFoundError(f"no saved artifact for model {model!r} for {city}")
    meta = max(cands, key=lambda m: m["version"])
    art = Path(meta["_path"]).with_suffix(".joblib")
    if meta.get("artifact_sha256") and _sha256_file(art) != meta["artifact_sha256"]:
        raise ValueError(f"{art.name} does not match the hash recorded by 04_modeling")
    pipe = joblib.load(art)
    cqr = None
    for p in sorted(models_dir.glob("cqr_*.joblib"), reverse=True):
        c = load_cqr(p)
        if c.get("split_version") == meta["split_version"] and c.get("features") == meta["features"]:
            cqr = c
            break
    if cqr is None:
        raise FileNotFoundError(f"no interval model (cqr_*.joblib) for {city} and this split")
    return pipe, cqr, meta, how


# ---------------------------------------------------------------------- saving
def save_opportunities(table: pd.DataFrame, listing_frame: pd.DataFrame, meta: dict,
                       reports_dir: Path | str) -> dict:
    """Write opportunities.csv (one row per segment), opportunity_listings.csv (the evidence,
    one row per listing) and opportunities.json (how the table was made)."""
    d = Path(reports_dir)
    d.mkdir(parents=True, exist_ok=True)
    version = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
    meta = {"table_version": version, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "segments": int(len(table)), "ranked_segments": int((table["status"] == "ok").sum()),
            "listings": int(len(listing_frame)), **meta}
    table.assign(table_version=version).to_csv(d / "opportunities.csv", index=False)
    listing_frame.assign(table_version=version).to_csv(d / "opportunity_listings.csv", index=False)
    (d / "opportunities.json").write_text(json.dumps(meta, indent=2, default=str))
    return meta


def load_opportunities(reports_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    d = Path(reports_dir)
    table = pd.read_csv(d / "opportunities.csv", dtype={"bedroom_band": str})
    table["rank"] = table["rank"].astype("Int64")
    listings = pd.read_csv(d / "opportunity_listings.csv", dtype={"bedroom_band": str})
    meta = json.loads((d / "opportunities.json").read_text())
    return table, listings, meta
