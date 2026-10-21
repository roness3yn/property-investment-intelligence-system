"""Target policy (design 3.2): which listings the price model is trained and judged on.

Rules
-----
* missing or non-positive price   -> excluded upstream (prepare_listings): no valid label
* extreme prices                  -> above the `price_upper_quantile` of the TRAINING partition
                                     (placeholder prices such as 9,999 are common)
* long-term rentals               -> minimum_nights above `max_minimum_nights` (monthly-stay
                                     pricing is a different market)
* inactive listings (optional)    -> no reviews in the last 12 months AND fully available

The thresholds are computed on the training partition only (`fit_target_policy`), stored in the
model metadata, and applied unchanged to calibration and test (`apply_target_policy`). Computing
them on all data would leak test information into the filter. Notebook 05 reports test MAE with
and without the excluded rows, so the filter cannot flatter the model.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULTS = {"price_upper_quantile": 0.995, "max_minimum_nights": 30, "exclude_inactive": False}


def fit_target_policy(train: pd.DataFrame, price_col: str = "price",
                      price_upper_quantile: float = DEFAULTS["price_upper_quantile"],
                      max_minimum_nights: int | None = DEFAULTS["max_minimum_nights"],
                      exclude_inactive: bool = DEFAULTS["exclude_inactive"]) -> dict:
    """Thresholds from the training partition only. Returns a JSON-serialisable dict."""
    prices = pd.to_numeric(train[price_col], errors="coerce").dropna()
    return {
        "price_upper_quantile": price_upper_quantile,
        "price_upper": float(np.quantile(prices, price_upper_quantile)),
        "max_minimum_nights": max_minimum_nights,
        "exclude_inactive": bool(exclude_inactive),
        "fitted_on": "training partition",
        "n_train_rows_used": int(len(prices)),
    }


def exclusion_reasons(df: pd.DataFrame, policy: dict, price_col: str = "price") -> pd.DataFrame:
    """One boolean column per rule: True where the row is excluded by that rule."""
    out = pd.DataFrame(index=df.index)
    price = pd.to_numeric(df[price_col], errors="coerce")
    out["price above cap"] = price > policy["price_upper"]
    if policy.get("max_minimum_nights") is not None and "minimum_nights" in df.columns:
        nights = pd.to_numeric(df["minimum_nights"], errors="coerce")
        out["long-term rental"] = nights > policy["max_minimum_nights"]
    if policy.get("exclude_inactive") and {"number_of_reviews_ltm", "availability_365"} <= set(df.columns):
        recent = pd.to_numeric(df["number_of_reviews_ltm"], errors="coerce")
        avail = pd.to_numeric(df["availability_365"], errors="coerce")
        out["inactive"] = (recent == 0) & (avail >= 365)
    return out


def apply_target_policy(df: pd.DataFrame, policy: dict, price_col: str = "price"
                        ) -> tuple[pd.DataFrame, pd.Series]:
    """(rows kept, per-reason counts of rows removed). A row can fail several rules."""
    reasons = exclusion_reasons(df, policy, price_col)
    drop = reasons.any(axis=1)
    counts = reasons.sum().astype(int)
    counts["total removed"] = int(drop.sum())
    counts["rows before"] = len(df)
    return df.loc[~drop].reset_index(drop=True), counts


def policy_scope_problem(row: pd.Series | dict, policy: dict | None) -> str | None:
    """For the prediction tool: why a listing is outside the population the model was
    trained on, or None. (The price cap concerns the label, so it is not checked here.)"""
    if not policy:
        return None
    limit = policy.get("max_minimum_nights")
    nights = pd.to_numeric(pd.Series([row.get("minimum_nights")]), errors="coerce").iloc[0]
    if limit is not None and pd.notna(nights) and nights > limit:
        return (f"minimum_nights {int(nights)} > {limit}: long-term rental, outside the "
                "model's training population")
    return None
