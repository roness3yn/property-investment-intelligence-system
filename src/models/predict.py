"""Read-only prediction tool for the agent (design 19.1).

The agent never loads model files itself. It calls one function with a listing id (or the
features of a hypothetical listing) and receives a fixed JSON-serialisable record:

    {
      "listing_id": 28871,
      "status": "ok",                      # ok | insufficient_evidence | not_found | invalid_input
      "status_reason": null,               # why, when status is not "ok"
      "currency": "EUR",
      "predicted_median_price": 182.0,     # null unless status == "ok"
      "interval_80": [141.0, 236.0],       # 80% conformal interval (null unless ok)
      "observed_price": 129.0,             # the listing's own asking price (null if unknown)
      "position": "below_interval",        # below_interval | within_interval | above_interval | null
      "in_training_data": false,           # true: the model saw this listing while training
      "top_drivers": [{"feature": "accommodates", "value": 4, "typical": 2,
                       "effect_pct": 21.4, "statement": "+21% for accommodates = 4 (typical: 2)"}],
      "drivers_method": "...",
      "assumptions": [],                   # values the tool filled in (hypothetical listings)
      "model_version": "hist_gbm_20261018_1406",
      "split_version": "2a418f50c8dc",
      "data_snapshot_date": "2026-06-15",
      "evidence_id": "pred:hist_gbm_20261018_1406:28871:2026-06-15"
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

Usage
-----
    from src.models.predict import PredictionTool
    tool = PredictionTool.load()                        # from the repository root
    tool.predict_listing(28871)
    tool.predict_features({"neighbourhood_cleansed": "Zuid", "room_type": "Entire home/apt",
                           "accommodates": 4, "bedrooms": 2})
    tool.handle_tool_call({"listing_id": 28871})        # what the agent framework calls
    TOOL_SCHEMA                                          # tool definition for the LLM

Command line (from the repository root):
    python -m src.models.predict --listing-id 28871
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

from src.models.data import ID_COL, SNAPSHOT_COL, TARGET, derive_bathrooms, prepare_listings
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
        "existing listing, or features for a hypothetical one. "
        "If status is not 'ok', do not state any price: report status_reason instead. "
        "The prediction is not a property value, purchase price, revenue or occupancy estimate. "
        "'below_interval' means the asking price is low compared with similar listings, not "
        "that the property is a good investment. top_drivers describe the model, not causes. "
        "Cite evidence_id for every claim based on this output."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "listing_id": {"type": "integer",
                           "description": "Inside Airbnb listing id of an existing listing."},
            "snapshot_date": {"type": "string",
                              "description": "Optional YYYY-MM-DD snapshot; latest if omitted."},
            "features": {
                "type": "object",
                "description": ("Hypothetical listing. Required: neighbourhood_cleansed, "
                                "room_type, accommodates. Optional: bedrooms, beds, "
                                "bathrooms or bathrooms_text, property_type, latitude, "
                                "longitude, minimum_nights and others."),
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
    def load(cls, repo_root: Path | str = ".", model: str | None = None,
             data_file: Path | str | None = None, limits: SupportLimits | None = None,
             evidence_log: Path | str | None = None, verbose: bool = True) -> "PredictionTool":
        """Load the selected model, its interval model and the listings data.

        model: model name ("hist_gbm", "ridge"). Default: the model selected in notebook 04
        (reports/model_card_metrics.json); otherwise notebook 03's development pick.
        data_file: processed listings file; default data/processed/<data_source in metadata>.
        evidence_log: optional .jsonl file; every response is appended (for groundedness checks).
        """
        root = Path(repo_root)
        models_dir = root / "models"
        metas = [json.loads(p.read_text()) | {"_path": str(p)} for p in models_dir.glob("*_*.json")]
        metas = [m for m in metas if "model" in m and "split_version" in m
                 and not m.get("synthetic_data", False)]
        if not metas:
            raise FileNotFoundError(f"no real-data model metadata in {models_dir}; run notebook 03")

        if model is None:
            card = root / "reports" / "model_card_metrics.json"
            selected = json.loads(card.read_text()).get("selected_model") if card.exists() else None
            if selected and not selected.startswith("B"):
                model = selected
            else:
                model = max(metas, key=lambda m: m["version"]).get("development_pick")
                warnings.warn(f"No model selected by notebook 04 yet: using 03's development "
                              f"pick {model!r}. Predictions are provisional.")
        candidates = [m for m in metas if m["model"] == model]
        if not candidates:
            raise FileNotFoundError(f"no saved artifact for model {model!r} in {models_dir}")
        meta = max(candidates, key=lambda m: m["version"])
        art_path = Path(meta["_path"]).with_suffix(".joblib")
        if meta.get("artifact_sha256") and _sha256_file(art_path) != meta["artifact_sha256"]:
            raise ValueError(f"{art_path.name} does not match the hash recorded by notebook 03")
        pipe = joblib.load(art_path)

        cqr = None
        for p in sorted(models_dir.glob("cqr_*.joblib"), reverse=True):
            c = load_cqr(p)
            if c.get("split_version") == meta["split_version"] and c.get("features") == meta["features"]:
                cqr = c
                break
        if cqr is None:
            raise FileNotFoundError("no interval model (cqr_*.joblib) for this split and feature "
                                    "list; run notebook 03 section 9")

        manifest = load_manifest(root / "evaluation" / "splits" / meta["split_file"])
        assert manifest["sha256"][:12] == meta["split_version"], "split file does not match model"

        data_path = Path(data_file) if data_file else root / "data" / "processed" / meta["data_source"]
        raw = select_city(ensure_snapshot_date(load_listings(data_path)), meta["city"], "location")
        listings = prepare_listings(raw, verbose=False, drop_missing_price=False)
        listings = apply_row_transforms(listings, load_audit(root / "evaluation" / "feature_audit.csv"))
        if meta.get("data_sha256") and _sha256_file(data_path) != meta["data_sha256"]:
            warnings.warn(f"{data_path.name} differs from the file the model was trained on "
                          "(new or reprocessed data). Predictions still work; support statistics "
                          "use the training listings found in this file.")

        tool = cls(pipe=pipe, cqr=cqr, meta=meta, listings=listings,
                   train_ids=set(manifest["train"]), limits=limits or SupportLimits(),
                   evidence_log=Path(evidence_log) if evidence_log else None)
        if verbose:
            print(f"{TOOL_NAME}: {tool.model_version}, split {meta['split_version']}, "
                  f"{meta['city']}, {len(listings):,} listings, intervals "
                  f"{int(round(100 * (1 - cqr['alpha'])))}%")
        return tool

    def __post_init__(self):
        self.features = list(self.meta["features"])
        self.currency = self.meta.get("currency")
        self.model_version = f"{self.meta['model']}_{self.meta['version']}"
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
                                            f"for {self.meta['city']}"))
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
        unknown = sorted(set(features) - set(self.features) - {"bathrooms_text"})
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
            "listing_id": _clean(listing_id), "status": status, "status_reason": reason,
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


def main() -> None:
    ap = argparse.ArgumentParser(description="Predict the nightly listing price of one listing.")
    ap.add_argument("--listing-id", type=int)
    ap.add_argument("--features", type=str, help='JSON, e.g. \'{"neighbourhood_cleansed": "Zuid", '
                                                  '"room_type": "Entire home/apt", "accommodates": 4}\'')
    ap.add_argument("--snapshot-date")
    ap.add_argument("--model", help="model name; default: selected in notebook 04")
    ap.add_argument("--repo-root", default=".")
    args = ap.parse_args()
    tool = PredictionTool.load(args.repo_root, model=args.model)
    call = ({"listing_id": args.listing_id, "snapshot_date": args.snapshot_date}
            if args.listing_id is not None else {"features": json.loads(args.features or "{}")})
    print(json.dumps(tool.handle_tool_call(call), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
