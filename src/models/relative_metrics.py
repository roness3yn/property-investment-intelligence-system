"""Relative (percentage) error measures for the nightly-price model.

MAE says how many euros a prediction is off; these measures say how far off it is relative to
the price, so "75 EUR" can be read against a 100 EUR room or a 500 EUR flat.

    typical_rel_err  exp(mean |ln(predicted / actual)|) - 1. Symmetric: being 20% too high and
                     20% too low count the same. The headline relative measure (also in
                     05_model_evaluation's tables).
    hit_rate_20      share of listings predicted within 20% of the actual price.
    MAPE             mean of |actual - predicted| / actual. Reported with a caveat: it explodes on
                     cheap listings (a 40 EUR room predicted at 100 EUR is a 150% error) and it
                     punishes over-prediction more than under-prediction, so a model must never
                     be tuned on it.
    MdAPE            median of |actual - predicted| / actual: the typical percentage error, robust
                     to the few cheap listings that dominate MAPE.

All values are returned in percent.

Command line (from the repository root), on the test predictions saved by 05_model_evaluation:

    python -m src.models.relative_metrics --predictions models/predictions_test_amsterdam_<split>.csv

It prints the measures for the selected model and the baseline (B1, refitted on the training
partition exactly as 05 does), overall and per price band, and saves both tables next to 05's
reports. It changes no model and no selection: it is reporting on predictions already made.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

MEASURES = ["MAE", "typical_rel_err", "hit_rate_20", "MAPE", "MdAPE"]
LABELS = {"MAE": "MAE (EUR/night)", "typical_rel_err": "typical relative error (%)",
          "hit_rate_20": "within ±20% (%)", "MAPE": "MAPE (%)", "MdAPE": "median APE (%)"}


def _arrays(y, pred) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    p = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(p) & (y > 0)
    return y[ok], np.clip(p[ok], 1e-6, None)        # a log ratio needs a positive prediction


def relative_metrics(y, pred) -> dict:
    """MAE in currency plus the four relative measures in percent (see the module docstring)."""
    y, p = _arrays(y, pred)
    if len(y) == 0:
        return {m: np.nan for m in MEASURES}
    ape = np.abs(y - p) / y
    return {"MAE": float(np.mean(np.abs(y - p))),
            "typical_rel_err": float(100 * (np.exp(np.mean(np.abs(np.log(p / y)))) - 1)),
            "hit_rate_20": float(100 * np.mean(ape <= 0.20)),
            "MAPE": float(100 * np.mean(ape)),
            "MdAPE": float(100 * np.median(ape))}


def by_price_band(y, preds: dict, n_bands: int = 10) -> pd.DataFrame:
    """The measures per band of the ACTUAL price (deciles by default), for each predictor.

    preds: {name: predictions}. Bands are defined by the actual price, so they answer
    "how wrong is the model for a 100 EUR listing versus a 500 EUR one?".
    """
    y = np.asarray(y, dtype=float)
    bands = pd.qcut(y, n_bands, labels=[f"D{i}" for i in range(1, n_bands + 1)])
    rows = []
    for band in bands.categories:
        m = np.asarray(bands == band)
        row = {"band": band, "listings": int(m.sum()),
               "price range (EUR)": f"{y[m].min():,.0f}–{y[m].max():,.0f}",
               "median price (EUR)": float(np.median(y[m]))}
        for name, p in preds.items():
            r = relative_metrics(y[m], np.asarray(p, dtype=float)[m])
            row[f"MAE {name}"] = r["MAE"]
            row[f"MdAPE {name} (%)"] = r["MdAPE"]
            row[f"typical rel. err. {name} (%)"] = r["typical_rel_err"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("band")


def summary_table(y, preds: dict) -> pd.DataFrame:
    """One row per predictor, one column per measure (readable column names)."""
    t = pd.DataFrame({name: relative_metrics(y, p) for name, p in preds.items()}).T[MEASURES]
    return t.rename(columns=LABELS)


# ---------------------------------------------------------------------- test-set report
def _baseline_on_test(root: Path, ids: pd.Series, city: str | None) -> pd.Series | None:
    """B1 refitted on the training partition (deterministic, as in 05) and predicted for `ids`."""
    from src.models.baselines import make_b1
    from src.models.data import ID_COL, TARGET, prepare_listings
    from src.models.make_split import ensure_snapshot_date, load_listings, load_manifest, select_city
    from src.models.target_policy import apply_target_policy

    metas = [json.loads(p.read_text()) for p in (root / "models").glob("*_*.json")]
    metas = [m for m in metas if "split_version" in m and not m.get("synthetic_data", False)
             and (city is None or m.get("city") == city)]
    if not metas:
        return None
    meta = max(metas, key=lambda m: m["version"])
    raw = ensure_snapshot_date(load_listings(root / "data" / "processed" / meta["data_source"]))
    df = prepare_listings(select_city(raw, meta["city"], "location"), verbose=False)
    manifest = load_manifest(root / "evaluation" / "splits" / meta["split_file"])
    train = df[df[ID_COL].isin(set(manifest["train"]))]
    train, _ = apply_target_policy(train, meta["target_policy"])
    b1 = make_b1().fit(train, train[TARGET])
    test = df[df[ID_COL].isin(set(ids))].drop_duplicates(ID_COL).set_index(ID_COL)
    pred = pd.Series(b1.predict(test.reset_index()), index=test.index)
    return pred.reindex(ids.to_numpy())


def main() -> None:
    ap = argparse.ArgumentParser(description="Relative error measures on 05's saved test predictions.")
    ap.add_argument("--predictions", required=True,
                    help="models/predictions_test_<city>_<split>.csv written by 05_model_evaluation")
    ap.add_argument("--root", default=".", help="repository root")
    ap.add_argument("--city", default=None)
    ap.add_argument("--no-baseline", action="store_true", help="skip refitting the baseline")
    a = ap.parse_args()
    root = Path(a.root)
    pt = pd.read_csv(a.predictions)
    y = pt["observed_price"].to_numpy()
    name = str(pt["model_version"].iloc[0]) if "model_version" in pt else "model"
    preds = {"model": pt["predicted_price"].to_numpy()}
    if not a.no_baseline:
        try:
            b1 = _baseline_on_test(root, pt["id"], a.city)
            if b1 is not None and b1.notna().all():
                preds = {"B1 baseline": b1.to_numpy(), **preds}
            else:
                print("baseline: not available (no model metadata or unmatched ids); model only")
        except Exception as e:                          # report the model even if B1 fails
            print(f"baseline skipped: {e}")
    pd.set_option("display.width", 200)
    print(f"test predictions: {len(pt):,} listings   model: {name}   (exploratory run, see README)\n")
    overall = summary_table(y, preds)
    print(overall.round(1).to_string(), "\n")
    bands = by_price_band(y, preds)
    print("By band of the actual price (D1 = cheapest 10%):")
    print(bands.round(1).to_string())
    stem = Path(a.predictions).stem                          # predictions_test_<city>_<split>
    parts = stem.split("_")
    city_dir = "_".join(parts[2:-1]) if stem.startswith("predictions_test_") and len(parts) > 3 else None
    if city_dir and not stem.endswith("_synthetic"):
        d = root / "reports" / city_dir
        d.mkdir(parents=True, exist_ok=True)
        overall.to_csv(d / "05_relative_metrics.csv")
        bands.to_csv(d / "05_relative_by_price_band.csv")
        print(f"\nsaved {d / '05_relative_metrics.csv'} and {d / '05_relative_by_price_band.csv'}")


if __name__ == "__main__":
    main()
