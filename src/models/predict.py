"""Read-only prediction tool for the agent (design 19.1).

The agent never loads model files itself. It calls one function with a listing id (or the
features of a hypothetical listing) and receives a fixed JSON-serialisable record:

    {
      "listing_id": 28871,
      "city": "Amsterdam",                 # one model per city; the answer says which
      "status": "ok",                      # ok | insufficient_evidence | not_found | invalid_input
      "status_reason": null,               # why, when status is not "ok"
      "currency": "EUR",                   # the city's local currency
      "predicted_median_price": 182.0,     # null unless status == "ok"
      "interval_80": [141.0, 236.0],       # 80% conformal interval (null unless ok)
      "observed_price": 129.0,             # the listing's own asking price (null if unknown)
      "position": "below_interval",        # below_interval | within_interval | above_interval | null
      "in_training_data": false,           # true: the model saw this listing while training
      "top_drivers": [{"feature": "accommodates", "value": 4, "typical": 2,
                       "effect_pct": 21.4, "statement": "+21% for accommodates = 4 (typical: 2)"}],
      "drivers_method": "...",
      "assumptions": [],                   # values the tool filled in (hypothetical listings)
      "model_version": "hist_gbm_amsterdam_20261018_1406",
      "split_version": "2a418f50c8dc",
      "data_snapshot_date": "2026-06-15",
      "evidence_id": "pred:hist_gbm_amsterdam_20261018_1406:28871:2026-06-15"
    }

Guardrails enforced in code, not in prompts:
* `insufficient_evidence` when the listing is outside the model's support: neighbourhood with
  too few training listings, room or property type never seen in training, coordinates outside
  the training area, missing core features, or an interval wider than the agreed limit. Prices
  are withheld in that case, so the agent cannot quote them.
* The prediction is a *listing* price per night, not an acquisition value; `position` compares
  the asking price with similar listings, it does not say the property is a good investment.
* `in_training_data` warns that the model has seen this listing, so its error there is
  optimistic and the interval's 80% guarantee does not apply.

Drivers. The design suggests SHAP; to avoid a new dependency, drivers are computed by
occlusion: each feature (or feature group) is replaced by its typical training value
(median, or most common category) and the change in the predicted price is reported. This is
model-agnostic, exact for what it states ("the prediction is X% higher than if this feature
were typical"), and has the same output shape, so SHAP can replace it later. Drivers describe
the model, not the market: they are not causal effects.

Several cities. There is one model per city (README). `PredictionService` loads a
`PredictionTool` for every city that has a trained model and routes each call: by `city` when
given, otherwise by looking the listing id up in each city's data. A hypothetical listing needs
`city` when more than one city is loaded.

Usage
-----
    from src.models.predict import PredictionService, TOOL_SCHEMA
    service = PredictionService.load()                   # from the repository root; all cities
    service.handle_tool_call({"listing_id": 28871})      # what the agent framework calls
    service.handle_tool_call({"city": "Amsterdam",
                              "features": {"neighbourhood_cleansed": "Zuid",
                                           "room_type": "Entire home/apt", "accommodates": 4}})
    TOOL_SCHEMA                                          # tool definition for the LLM

    from src.models.predict import PredictionTool        # one city only
    tool = PredictionTool.load(city="Amsterdam")

Command line (from the repository root):
    python -m src.models.predict --listing-id 28871
    python -m src.models.predict --city Amsterdam --features "{...}"
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.models.cities import city_data_sha256, city_slug
from src.models.data import ID_COL, SNAPSHOT_COL, TARGET, derive_bathrooms, prepare_listings
from src.models.engineered import (AMENITY_FEATURES, COMPETITION_FEATURES, NEIGHBOURHOOD_FEATURES,
                                   CompetitionIndex, amenity_features, neighbourhood_counts)
from src.models.feature_audit import apply_row_transforms, load_audit
from src.models.intervals import load_cqr, predict_interval
from src.models.make_split import ensure_snapshot_date, load_listings, load_manifest, select_city

TOOL_NAME = "predict_listing_price"
CORE_FEATURES = ["neighbourhood_cleansed", "room_type", "accommodates", "latitude", "longitude"]
# Features reported together as one driver
DRIVER_GROUPS = {
    "location (coordinates)": ["latitude", "longitude"],
    "review scores": ["review_scores_rating", "review_scores_accuracy",
                      "review_scores_cleanliness", "review_scores_checkin",
                      "review_scores_communication", "review_scores_location",
                      "review_scores_value"],
}


@dataclass
class SupportLimits:
    """When the tool refuses to answer (design 19.1, "Watch out"). Agree these with the team."""
    min_neighbourhood_listings: int = 20      # training listings in the neighbourhood
    max_relative_width: float = 2.0           # (upper - lower) / predicted
    coordinate_margin_deg: float = 0.02       # allowed distance outside the training area


TOOL_SCHEMA = {
    "name": TOOL_NAME,
    "description": (
        "Predicts the typical nightly LISTING price (local currency per night) for a short-term "
        "rental listing, with an 80% interval, and says whether the listing's own asking price "
        "is below, within or above that interval. Read-only. Use either listing_id for an "
        "existing listing, or features (plus city) for a hypothetical one. There is one model per "
        "city; prices are in that city's local currency, given in `currency`: never compare or "
        "average prices across currencies. "
        "If status is not 'ok', do not state any price: report status_reason instead. "
        "The prediction is not a property value, purchase price, revenue or occupancy estimate. "
        "'below_interval' means the asking price is low compared with similar listings, not "
        "that the property is a good investment. top_drivers describe the model, not causes. "
        "Cite evidence_id for every claim based on this output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string",
                     "description": ("City, e.g. 'Amsterdam'. Optional with listing_id; required "
                                     "with features when several cities are available.")},
            "listing_id": {"type": "integer",
                           "description": "Inside Airbnb listing id of an existing listing."},
            "snapshot_date": {"type": "string",
                              "description": "Optional YYYY-MM-DD snapshot; latest if omitted."},
            "features": {
                "type": "object",
                "description": ("Hypothetical listing. Required: neighbourhood_cleansed, "
                                "room_type, accommodates. Optional: bedrooms, beds, "
                                "bathrooms or bathrooms_text, property_type, latitude, "
                                "longitude, minimum_nights, amenities (list of strings, "
                                "e.g. [\"Wifi\", \"Kitchen\"]) and others. Competitor and "
                                "neighbourhood counts are computed by the tool."),
            },
        },
    },
}


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _bare_result(status: str, reason: str, listing_id=None, city=None) -> dict:
    """Contract-shaped answer for errors raised before a city's model is chosen."""
    return {"listing_id": _clean(listing_id), "city": city, "status": status,
            "status_reason": reason, "currency": None, "predicted_median_price": None,
            "interval_80": None, "observed_price": None, "position": None,
            "in_training_data": None, "top_drivers": [], "drivers_method": None,
            "assumptions": [], "model_version": None, "split_version": None,
            "data_snapshot_date": None,
            "evidence_id": f"pred:none:{_clean(listing_id) if listing_id is not None else 'na'}:na"}


def _real_metas(models_dir: Path) -> list[dict]:
    metas = [json.loads(p.read_text()) | {"_path": str(p)} for p in models_dir.glob("*_*.json")]
    return [m for m in metas if "model" in m and "split_version" in m and "city" in m
            and not m.get("synthetic_data", False)]


def _selected_model(root: Path, city: str) -> str | None:
    """The model 05_model_evaluation selected for this city (reports/<city>/model_card_metrics.json)."""
    for card in [root / "reports" / city_slug(city) / "model_card_metrics.json",
                 root / "reports" / "model_card_metrics.json"]:          # before multi-city
        if card.exists():
            info = json.loads(card.read_text())
            if info.get("city") == city and not str(info.get("label", "")).startswith("SYNTHETIC"):
                sel = info.get("selected_model")
                return sel if sel and not sel.startswith("B") else None
    return None


def _clean(v):
    """JSON-friendly scalar: numpy -> python, NaN -> None, round floats."""
    if v is None:
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return None if math.isnan(v) else round(float(v), 2)
    if isinstance(v, (pd.Timestamp, datetime)):
        return v.strftime("%Y-%m-%d")
    return v


def _fmt(v):
    v = _clean(v)
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v)


@dataclass
class PredictionTool:
    pipe: object
    cqr: dict
    meta: dict
    listings: pd.DataFrame            # prepared listings (all snapshots, price may be NaN)
    train_ids: set
    limits: SupportLimits = field(default_factory=SupportLimits)
    evidence_log: Path | None = None

    # ------------------------------------------------------------------ loading
    @classmethod
    def load(cls, repo_root: Path | str = ".", city: str | None = None, model: str | None = None,
             data_file: Path | str | None = None, raw_listings: pd.DataFrame | None = None,
             limits: SupportLimits | None = None, evidence_log: Path | str | None = None,
             verbose: bool = True) -> "PredictionTool":
        """Load one city's selected model, its interval model and its listings.

        city: required when models exist for several cities (or use PredictionService).
        model: model name ("hist_gbm", "ridge"). Default: the model 05_model_evaluation selected for
        this city (reports/<city>/model_card_metrics.json); otherwise 04_modeling's development pick.
        data_file: processed listings file; default data/processed/<data_source in metadata>.
        raw_listings: the already-loaded listings file (all cities), to avoid re-reading it.
        evidence_log: optional .jsonl file; every response is appended (for groundedness checks).
        """
        root = Path(repo_root)
        models_dir = root / "models"
        metas = _real_metas(models_dir)
        if not metas:
            raise FileNotFoundError(f"no real-data model metadata in {models_dir}; run 04_modeling")
        cities = sorted({m["city"] for m in metas})
        if city is None:
            if len(cities) > 1:
                raise ValueError(f"models exist for several cities {cities}: pass city=..., or "
                                 "use PredictionService")
            city = cities[0]
        metas = [m for m in metas if m["city"] == city]
        if not metas:
            raise FileNotFoundError(f"no trained model for {city!r}; available: {cities}")

        if model is None:
            model = _selected_model(root, city)
            if model is None:
                model = max(metas, key=lambda m: m["version"]).get("development_pick")
                warnings.warn(f"{city}: no model selected by 05_model_evaluation yet; using 04_modeling's "
                              f"development pick {model!r}. Predictions are provisional.")
        candidates = [m for m in metas if m["model"] == model]
        if not candidates:
            raise FileNotFoundError(f"no saved artifact for model {model!r} for {city}")
        meta = max(candidates, key=lambda m: m["version"])
        art_path = Path(meta["_path"]).with_suffix(".joblib")
        if meta.get("artifact_sha256") and _sha256_file(art_path) != meta["artifact_sha256"]:
            raise ValueError(f"{art_path.name} does not match the hash recorded by 04_modeling")
        pipe = joblib.load(art_path)

        cqr = None
        for p in sorted(models_dir.glob("cqr_*.joblib"), reverse=True):
            c = load_cqr(p)
            if (c.get("split_version") == meta["split_version"]
                    and c.get("features") == meta["features"] and c.get("city", city) == city):
                cqr = c
                break
        if cqr is None:
            raise FileNotFoundError(f"no interval model (cqr_*.joblib) for {city}, this split and "
                                    "feature list; run 04_modeling section 9")

        manifest = load_manifest(root / "evaluation" / "splits" / meta["split_file"])
        assert manifest["sha256"][:12] == meta["split_version"], "split file does not match model"

        if raw_listings is None:
            data_path = Path(data_file) if data_file else root / "data" / "processed" / meta["data_source"]
            raw_listings = ensure_snapshot_date(load_listings(data_path))
        raw = select_city(raw_listings, city, "location")
        if meta.get("data_sha256") and city_data_sha256(raw) != meta["data_sha256"]:
            warnings.warn(f"{city}: the listings differ from those the model was trained on "
                          "(new or reprocessed data). Predictions still work; support statistics "
                          "use the training listings found in the current data.")
        listings = prepare_listings(raw, verbose=False, drop_missing_price=False)
        listings = apply_row_transforms(listings, load_audit(root / "evaluation" / "feature_audit.csv"))

        tool = cls(pipe=pipe, cqr=cqr, meta=meta, listings=listings,
                   train_ids=set(manifest["train"]), limits=limits or SupportLimits(),
                   evidence_log=Path(evidence_log) if evidence_log else None)
        if verbose:
            print(f"{TOOL_NAME}: {city}: {tool.model_version}, split {meta['split_version']}, "
                  f"{len(listings):,} listings, {tool.currency}, intervals "
                  f"{int(round(100 * (1 - cqr['alpha'])))}%")
        return tool

    def __post_init__(self):
        self.features = list(self.meta["features"])
        self.city = self.meta["city"]
        self.currency = self.meta.get("currency")
        slug = self.meta.get("city_slug")
        self.model_version = "_".join(p for p in [self.meta["model"], slug, self.meta["version"]] if p)
        self.listing_ids = set(self.listings[ID_COL])
        train = self.listings[self.listings[ID_COL].isin(self.train_ids)]
        if train.empty:
            raise ValueError("no training listings found in the data file")
        one = train.sort_values(SNAPSHOT_COL).groupby(ID_COL).tail(1)
        self._categorical = [c for c in self.features
                             if not pd.api.types.is_numeric_dtype(train[c])]
        self._hood_counts = one["neighbourhood_cleansed"].value_counts()
        self._seen = {c: set(one[c].dropna().astype(str))
                      for c in ["room_type", "property_type"] if c in self.features}
        self._bounds = {c: (one[c].min(), one[c].max()) for c in ["latitude", "longitude"]}
        self._centroids = one.groupby("neighbourhood_cleansed")[["latitude", "longitude"]].median()
        # Engineered features (03_Feature_Engineering) recomputed for hypothetical listings,
        # over the city's current listings (latest snapshot), as that notebook defines them
        latest = self.listings[self.listings[SNAPSHOT_COL] == self.listings[SNAPSHOT_COL].max()]
        self._competition = (CompetitionIndex(latest)
                             if set(COMPETITION_FEATURES) & set(self.features) else None)
        hood_feats = [c for c in NEIGHBOURHOOD_FEATURES if c in self.features]
        if not hood_feats:
            self._hood_table = None
        elif set(hood_feats) <= set(latest.columns):    # use the file's own values
            self._hood_table = latest.groupby("neighbourhood_cleansed")[hood_feats].first()
        else:
            self._hood_table = neighbourhood_counts(latest)
        # Typical values for the occlusion drivers
        self._typical = {}
        for c in self.features:
            s = train[c]
            if pd.api.types.is_numeric_dtype(s):
                self._typical[c] = float(s.median()) if s.notna().any() else np.nan
            else:
                self._typical[c] = s.mode().iloc[0] if s.notna().any() else np.nan

    # ------------------------------------------------------------------ public API
    def predict_listing(self, listing_id: int, snapshot_date: str | None = None) -> dict:
        rows = self.listings[self.listings[ID_COL] == listing_id]
        if snapshot_date is not None:
            rows = rows[rows[SNAPSHOT_COL] == pd.Timestamp(snapshot_date)]
        if rows.empty:
            when = f" in snapshot {snapshot_date}" if snapshot_date else ""
            return self._finish(self._empty(listing_id, snapshot_date, "not_found",
                                            f"listing {listing_id} not found{when} "
                                            f"for {self.city}"))
        row = rows.sort_values(SNAPSHOT_COL).iloc[[-1]].reset_index(drop=True)
        return self._finish(self._predict_row(row, listing_id=int(listing_id),
                                              observed=row[TARGET].iloc[0],
                                              snapshot=row[SNAPSHOT_COL].iloc[0], assumptions=[]))

    def predict_features(self, features: dict) -> dict:
        """Hypothetical listing. Missing coordinates are filled with the neighbourhood's
        typical location; other missing values are imputed by the model as in training."""
        required = ["neighbourhood_cleansed", "room_type", "accommodates"]
        missing = [k for k in required if features.get(k) in (None, "")]
        if missing:
            return self._finish(self._empty(None, None, "invalid_input",
                                            f"missing required features: {missing}"))
        unknown = sorted(set(features) - set(self.features) - {"bathrooms_text", "amenities"})
        row = pd.DataFrame([{c: features.get(c, np.nan) for c in self.features}])
        assumptions = []
        if unknown:
            assumptions.append(f"ignored unknown fields: {unknown}")
        if "bathrooms_text" in features and pd.isna(row.get("bathrooms", pd.Series([np.nan]))[0]):
            row = derive_bathrooms(row.assign(bathrooms_text=features["bathrooms_text"]))
        hood = features["neighbourhood_cleansed"]
        if (pd.isna(row["latitude"][0]) or pd.isna(row["longitude"][0])) and hood in self._centroids.index:
            row[["latitude", "longitude"]] = self._centroids.loc[hood].to_numpy()
            assumptions.append(f"coordinates set to the typical location in {hood}")
        assumptions += self._fill_engineered(row, features)
        if "host_is_superhost" in row and isinstance(row["host_is_superhost"][0], str):
            row["host_is_superhost"] = row["host_is_superhost"].map({"t": 1.0, "f": 0.0})
        for c in self.features:
            if c not in self._categorical:
                row[c] = pd.to_numeric(row[c], errors="coerce")
        imputed = [c for c in self.features if pd.isna(row[c][0])]
        if imputed:
            assumptions.append(f"not provided, imputed by the model: {imputed}")
        return self._finish(self._predict_row(row, listing_id=None, observed=np.nan,
                                              snapshot=None, assumptions=assumptions))

    def handle_tool_call(self, tool_input: dict) -> dict:
        """Entry point for the agent framework: validates input, never raises."""
        try:
            if tool_input.get("city") not in (None, "", self.city):
                return self._finish(self._empty(tool_input.get("listing_id"), None, "invalid_input",
                                                f"this tool serves {self.city}, not "
                                                f"{tool_input['city']!r}"))
            if tool_input.get("listing_id") is not None:
                return self.predict_listing(int(tool_input["listing_id"]),
                                            tool_input.get("snapshot_date"))
            if isinstance(tool_input.get("features"), dict):
                return self.predict_features(tool_input["features"])
            return self._finish(self._empty(None, None, "invalid_input",
                                            "provide listing_id or features"))
        except Exception as exc:                                  # tool errors are data
            return self._finish(self._empty(tool_input.get("listing_id"), None, "invalid_input",
                                            f"{type(exc).__name__}: {exc}"))

    def predict_many(self, listing_ids) -> pd.DataFrame:
        """Convenience for reports: one row per listing (drivers omitted)."""
        out = [self.predict_listing(int(i)) for i in listing_ids]
        df = pd.DataFrame(out).drop(columns=["top_drivers", "drivers_method", "assumptions"])
        df[["interval_lo", "interval_hi"]] = pd.DataFrame(
            [r if r else [None, None] for r in df.pop("interval_80")], index=df.index)
        return df

    # ------------------------------------------------------------------ internals
    def _fill_engineered(self, row: pd.DataFrame, features: dict) -> list[str]:
        """Compute 03_Feature_Engineering's columns the model uses but the caller did not give."""
        notes = []
        wanted = lambda cols: [c for c in cols if c in self.features and features.get(c) is None]
        amen = wanted(AMENITY_FEATURES)
        if amen and features.get("amenities") is not None:
            values = amenity_features(features["amenities"])
            for c in amen:
                row[c] = values[c]
            notes.append("amenity features computed from the amenities list")
        comp = wanted(COMPETITION_FEATURES)
        lat, lon = row["latitude"][0], row["longitude"][0]
        if comp and self._competition is not None and pd.notna(lat) and pd.notna(lon):
            values = self._competition.counts(lat, lon, row["room_type"][0])
            for c in comp:
                row[c] = values[c]
            notes.append(f"competitor counts computed from the coordinates over {self.city}'s "
                         "current listings")
        hood_cols = wanted(NEIGHBOURHOOD_FEATURES)
        hood = row["neighbourhood_cleansed"][0]
        if hood_cols and self._hood_table is not None and hood in self._hood_table.index:
            for c in hood_cols:
                row[c] = self._hood_table.loc[hood, c]
            notes.append(f"neighbourhood counts taken from {hood}")
        return notes

    def _support_problems(self, row: pd.DataFrame) -> list[str]:
        r = row.iloc[0]
        problems = []
        miss = [c for c in CORE_FEATURES if c in row and pd.isna(r[c])]
        if miss:
            problems.append(f"missing core features: {miss}")
        hood = r["neighbourhood_cleansed"]
        n = int(self._hood_counts.get(hood, 0))
        if n < self.limits.min_neighbourhood_listings:
            problems.append(f"neighbourhood {hood!r} has {n} training listings "
                            f"(< {self.limits.min_neighbourhood_listings})")
        for c, seen in self._seen.items():
            if pd.notna(r[c]) and str(r[c]) not in seen:
                problems.append(f"{c} {r[c]!r} never seen in training")
        m = self.limits.coordinate_margin_deg
        for c, (lo, hi) in self._bounds.items():
            if pd.notna(r[c]) and not (lo - m <= r[c] <= hi + m):
                problems.append(f"{c} {r[c]:.4f} outside the training area")
        return problems

    def _drivers(self, row: pd.DataFrame, base_log: float) -> list[dict]:
        grouped = {c for cols in DRIVER_GROUPS.values() for c in cols}
        groups = {k: [c for c in v if c in self.features] for k, v in DRIVER_GROUPS.items()}
        groups.update({c: [c] for c in self.features if c not in grouped})
        groups = {k: v for k, v in groups.items() if v}

        variants, labels = [], []
        for label, cols in groups.items():
            vals = row.iloc[0][cols]
            if vals.isna().all():
                continue                      # missing: the model already uses a typical value
            typical = {c: self._typical[c] for c in cols}
            if all(_clean(vals[c]) == _clean(typical[c]) for c in cols):
                continue
            v = row.copy()
            for c in cols:
                v[c] = typical[c]
            variants.append(v)
            labels.append(label)
        if not variants:
            return []
        alt = self.pipe.predict(pd.concat(variants, ignore_index=True)[self.features])
        drivers = []
        for label, a in zip(labels, alt):
            effect = 100 * (math.exp(base_log - a) - 1)
            cols = groups[label]
            value = _clean(row.iloc[0][cols[0]]) if len(cols) == 1 else "listing's own values"
            typical = _clean(self._typical[cols[0]]) if len(cols) == 1 else "typical values"
            name = label if len(cols) > 1 else cols[0]
            statement = (f"{effect:+.0f}% for {name} = {_fmt(value)} (typical: {_fmt(typical)})"
                         if len(cols) == 1 else f"{effect:+.0f}% for {name} vs typical")
            drivers.append({"feature": name, "value": value, "typical": typical,
                            "effect_pct": round(effect, 1), "statement": statement})
        up = sorted([d for d in drivers if d["effect_pct"] > 0], key=lambda d: -d["effect_pct"])[:3]
        down = sorted([d for d in drivers if d["effect_pct"] < 0], key=lambda d: d["effect_pct"])[:3]
        return [d for d in up + down if abs(d["effect_pct"]) >= 0.5]

    def _predict_row(self, row, listing_id, observed, snapshot, assumptions) -> dict:
        row = row[self.features].copy()
        out = self._empty(listing_id, snapshot, "ok", None)
        out["assumptions"] = assumptions
        out["observed_price"] = _clean(observed)
        out["in_training_data"] = bool(listing_id in self.train_ids) if listing_id is not None else False

        problems = self._support_problems(row)
        base_log = float(self.pipe.predict(row)[0])
        pred = math.expm1(base_log) if self.meta.get("log_target", True) else base_log
        lo, hi = (float(a[0]) for a in predict_interval(self.cqr, row))
        if pred > 0 and (hi - lo) / pred > self.limits.max_relative_width:
            problems.append(f"interval too wide ({(hi - lo) / pred:.1f}x the prediction; "
                            f"limit {self.limits.max_relative_width})")
        if problems:
            out["status"], out["status_reason"] = "insufficient_evidence", "; ".join(problems)
            return out

        out["predicted_median_price"] = round(pred, 2)
        out["interval_80"] = [round(lo, 2), round(hi, 2)]
        if out["observed_price"] is not None:
            y = out["observed_price"]
            out["position"] = ("below_interval" if y < lo else
                               "above_interval" if y > hi else "within_interval")
        out["top_drivers"] = self._drivers(row, base_log)
        return out

    def _empty(self, listing_id, snapshot, status, reason) -> dict:
        snap = _clean(snapshot)
        return {
            "listing_id": _clean(listing_id), "city": self.city, "status": status,
            "status_reason": reason,
            "currency": self.currency, "predicted_median_price": None, "interval_80": None,
            "observed_price": None, "position": None, "in_training_data": None,
            "top_drivers": [],
            "drivers_method": ("occlusion: change in the prediction when the feature is set to "
                               "its typical training value; describes the model, not causes"),
            "assumptions": [], "model_version": self.model_version,
            "split_version": self.meta["split_version"], "data_snapshot_date": snap,
            "evidence_id": None,
        }

    def _finish(self, out: dict) -> dict:
        lid = out["listing_id"] if out["listing_id"] is not None else "hypothetical"
        out["evidence_id"] = f"pred:{self.model_version}:{lid}:{out['data_snapshot_date'] or 'na'}"
        if out["listing_id"] is None and out["status"] == "ok":
            # hypothetical inputs differ per call: make the id unique and reproducible
            digest = hashlib.sha1(json.dumps(out["assumptions"] + [out["predicted_median_price"]],
                                             default=str).encode()).hexdigest()[:8]
            out["evidence_id"] += f":{digest}"
        if self.evidence_log is not None:
            self.evidence_log.parent.mkdir(parents=True, exist_ok=True)
            record = {"logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **out}
            with open(self.evidence_log, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        return out


class PredictionService:
    """All cities behind one tool: routes each call to that city's PredictionTool."""

    def __init__(self, tools: dict[str, PredictionTool]):
        if not tools:
            raise ValueError("no city tools to serve")
        self.tools = tools

    @classmethod
    def load(cls, repo_root: Path | str = ".", cities=None, data_file: Path | str | None = None,
             limits: SupportLimits | None = None, evidence_log: Path | str | None = None,
             verbose: bool = True) -> "PredictionService":
        """One tool per city with a trained real-data model (or only `cities`). The listings
        file is read once and shared."""
        root = Path(repo_root)
        metas = _real_metas(root / "models")
        available = sorted({m["city"] for m in metas})
        if not available:
            raise FileNotFoundError(f"no real-data model metadata in {root / 'models'}; run 04_modeling")
        cities = list(cities) if cities else available
        missing = sorted(set(cities) - set(available))
        if missing:
            raise FileNotFoundError(f"no trained model for {missing}; available: {available}")
        sources = {m["data_source"] for m in metas if m["city"] in cities}
        if data_file is None and len(sources) > 1:
            raise ValueError(f"cities were trained on different files {sorted(sources)}: pass data_file")
        data_path = Path(data_file) if data_file else root / "data" / "processed" / sources.pop()
        raw = ensure_snapshot_date(load_listings(data_path))
        return cls({c: PredictionTool.load(root, city=c, raw_listings=raw, limits=limits,
                                           evidence_log=evidence_log, verbose=verbose)
                    for c in cities})

    @property
    def cities(self) -> list[str]:
        return sorted(self.tools)

    def handle_tool_call(self, tool_input: dict) -> dict:
        """Entry point for the agent framework: routes by city or listing id, never raises."""
        if not isinstance(tool_input, dict):
            return _bare_result("invalid_input", "tool input must be an object")
        city = tool_input.get("city") or None
        lid = tool_input.get("listing_id")
        if city is not None:
            if city not in self.tools:
                return _bare_result("invalid_input", f"no model for city {city!r}; available: "
                                    f"{self.cities}", lid, city)
            return self.tools[city].handle_tool_call(tool_input)
        if lid is not None:
            try:
                lid_int = int(lid)
            except (TypeError, ValueError):
                return _bare_result("invalid_input", f"listing_id must be an integer, got {lid!r}", lid)
            hits = [c for c, t in self.tools.items() if lid_int in t.listing_ids]
            if len(hits) == 1:
                return self.tools[hits[0]].handle_tool_call(tool_input)
            if not hits:
                return _bare_result("not_found", f"listing {lid_int} not found in any modelled "
                                    f"city ({', '.join(self.cities)})", lid_int)
            return _bare_result("invalid_input", f"listing {lid_int} exists in several cities "
                                f"{hits}: pass city", lid_int)
        if len(self.tools) == 1:
            return next(iter(self.tools.values())).handle_tool_call(tool_input)
        return _bare_result("invalid_input", "pass city for a hypothetical listing "
                            f"(available: {', '.join(self.cities)})")

    def predict_listing(self, listing_id: int, city: str | None = None,
                        snapshot_date: str | None = None) -> dict:
        return self.handle_tool_call({"listing_id": listing_id, "city": city,
                                      "snapshot_date": snapshot_date})


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict the nightly listing price of one listing.")
    ap.add_argument("--listing-id", type=int)
    ap.add_argument("--city", help="e.g. Amsterdam; needed with --features if several cities exist")
    ap.add_argument("--features", type=str, help='JSON, e.g. \'{"neighbourhood_cleansed": "Zuid", '
                                                  '"room_type": "Entire home/apt", "accommodates": 4}\'')
    ap.add_argument("--snapshot-date")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    service = PredictionService.load(args.repo_root, cities=[args.city] if args.city else None)
    call = {"city": args.city}
    call.update({"listing_id": args.listing_id, "snapshot_date": args.snapshot_date}
                if args.listing_id is not None else {"features": json.loads(args.features or "{}")})
    print(json.dumps(service.handle_tool_call(call), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
