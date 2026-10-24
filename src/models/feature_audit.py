"""Feature leakage audit: the feature list as data, not code.

Design reference: Modelling & Evaluation Design v2, section 3.3, and blueprint section 5.

Every candidate column is one row of evaluation/feature_audit.csv, classified on two axes:

* availability  - would the value be known for a new listing at prediction time?
* endogeneity   - is the value itself partly caused by the price?

and given a decision. Only rows with decision == "include" can become model features.

Decisions
---------
include     used by the main model (if the column exists in the data)
exclude     never used (target leakage or endogenous)
ablation    kept out of the main model; tested only in a dedicated ablation
defer       not used yet (e.g. free text); revisit with an ablation
planned     approved but the transform is not implemented yet (e.g. amenities parsing)
source      raw column used only to derive other features (e.g. bathrooms_text)
identifier  keys and dates (id, snapshot_date, scrape_id): never features
target      the prediction target

Transforms (column "transform")
-------------------------------
missing_indicator  numeric: median-impute and add an explicit "was missing" flag
collapse_rare      categorical: rare levels are grouped (OneHotEncoder min_frequency)
tf_to_binary       Inside Airbnb "t"/"f" strings -> 1/0 (row-wise, no fitting)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DECISIONS = {"include", "exclude", "ablation", "defer", "planned", "source", "identifier", "target"}
TYPES = {"numeric", "categorical", "binary", "text", "date", "list"}
TRANSFORMS = {"", "missing_indicator", "collapse_rare", "tf_to_binary"}
REQUIRED_COLUMNS = ["feature", "group", "type", "available_at_prediction", "endogenous",
                    "decision", "transform", "in_dictionary", "derived_from", "notes"]


def load_audit(path: Path | str) -> pd.DataFrame:
    """Read and validate the audit table. Fails loudly on a malformed file."""
    audit = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = [c for c in REQUIRED_COLUMNS if c not in audit.columns]
    assert not missing, f"feature_audit.csv is missing columns: {missing}"
    audit = audit[REQUIRED_COLUMNS].apply(lambda s: s.str.strip())

    dupes = audit["feature"][audit["feature"].duplicated()].tolist()
    assert not dupes, f"duplicate features in audit: {dupes}"
    bad = audit.loc[~audit["decision"].isin(DECISIONS), "feature"].tolist()
    assert not bad, f"unknown decision for: {bad} (allowed: {sorted(DECISIONS)})"
    bad = audit.loc[~audit["type"].isin(TYPES), "feature"].tolist()
    assert not bad, f"unknown type for: {bad} (allowed: {sorted(TYPES)})"
    bad = audit.loc[~audit["transform"].isin(TRANSFORMS), "feature"].tolist()
    assert not bad, f"unknown transform for: {bad} (allowed: {sorted(TRANSFORMS - {''})})"
    bad = audit.loc[~audit["in_dictionary"].isin({"yes", "no"}), "feature"].tolist()
    assert not bad, f"in_dictionary must be yes/no for: {bad}"
    no_reason = audit.loc[audit["decision"].isin({"exclude", "ablation", "defer"})
                          & (audit["notes"] == ""), "feature"].tolist()
    assert not no_reason, f"exclusions need a reason in 'notes': {no_reason}"
    assert (audit["decision"] == "target").sum() == 1, "exactly one row must be the target"
    return audit


def apply_row_transforms(df: pd.DataFrame, audit: pd.DataFrame) -> pd.DataFrame:
    """Row-wise conversions that need no fitting (safe before the split)."""
    df = df.copy()
    for feat in audit.loc[audit["transform"] == "tf_to_binary", "feature"]:
        if feat in df.columns and not pd.api.types.is_numeric_dtype(df[feat]):
            df[feat] = df[feat].map({"t": 1, "f": 0, True: 1, False: 0}).astype(float)
    return df


@dataclass
class FeatureSet:
    numeric: list = field(default_factory=list)            # median-imputed
    numeric_indicator: list = field(default_factory=list)  # median-imputed + missing flag
    categorical: list = field(default_factory=list)
    not_in_dictionary: list = field(default_factory=list)  # used, but must be added to the dictionary
    skipped_missing: list = field(default_factory=list)    # included in audit, absent from data
    skipped_empty: list = field(default_factory=list)      # included in audit, 100% missing in data
    blocked_present: dict = field(default_factory=dict)    # non-include columns found in data
    planned: list = field(default_factory=list)
    unaudited: list = field(default_factory=list)          # data columns the audit does not mention

    @property
    def all(self) -> list:
        return self.numeric + self.numeric_indicator + self.categorical


def select_features(audit: pd.DataFrame, columns, strict: bool = False,
                    empty_columns=()) -> FeatureSet:
    """Build the feature lists from rows marked include, checked against the data columns.

    empty_columns: columns present but 100% missing (e.g. a field Inside Airbnb stopped
    filling). They are skipped like absent ones, except core dictionary features, which fail.
    """
    columns = list(columns)
    empty_columns = set(empty_columns)
    fs = FeatureSet()
    for row in audit.itertuples(index=False):
        present = row.feature in columns
        if row.decision == "include":
            if not present:
                if row.in_dictionary == "yes":
                    raise AssertionError(f"core feature '{row.feature}' (in the data dictionary) "
                                         "is missing from the data")
                fs.skipped_missing.append(row.feature)
                continue
            if row.feature in empty_columns:
                if row.in_dictionary == "yes":
                    raise AssertionError(f"core feature '{row.feature}' is entirely missing")
                fs.skipped_empty.append(row.feature)
                continue
            if row.type == "categorical":
                fs.categorical.append(row.feature)
            elif row.transform == "missing_indicator":
                fs.numeric_indicator.append(row.feature)
            else:                                   # numeric and binary
                fs.numeric.append(row.feature)
            if row.in_dictionary == "no":
                fs.not_in_dictionary.append(row.feature)
        else:
            if row.decision == "planned":
                fs.planned.append(row.feature)
            if present:
                fs.blocked_present[row.feature] = row.decision

    blocked = set(audit.loc[audit["decision"] != "include", "feature"])
    leaked = blocked & set(fs.all)
    assert not leaked, f"non-include features in the feature list: {sorted(leaked)}"

    fs.unaudited = sorted(set(columns) - set(audit["feature"]))
    if fs.unaudited and strict:
        raise AssertionError(f"{len(fs.unaudited)} data columns are not in the audit: "
                             f"{fs.unaudited[:20]}")
    return fs


def describe(fs: FeatureSet) -> pd.DataFrame:
    """One-line-per-category summary for display in the notebook."""
    rows = {
        "numeric (median impute)": fs.numeric,
        "numeric (+ missing indicator)": fs.numeric_indicator,
        "categorical": fs.categorical,
        "used but not in data dictionary": fs.not_in_dictionary,
        "include in audit, absent from data": fs.skipped_missing,
        "include in audit, 100% missing in data": fs.skipped_empty,
        "planned (not implemented)": fs.planned,
        "present but blocked": [f"{k} ({v})" for k, v in fs.blocked_present.items()],
        "in data, not in audit": fs.unaudited,
    }
    return pd.DataFrame({"n": {k: len(v) for k, v in rows.items()},
                         "columns": {k: ", ".join(v) if v else "-" for k, v in rows.items()}})
