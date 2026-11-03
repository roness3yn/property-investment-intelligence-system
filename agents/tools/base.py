"""Tool specification and the shared, lazily loaded data context.

A tool is a read-only Python function the language model can call. Each tool has:

* `schema`: the name, description and JSON input schema sent to the model;
* `handler(args) -> dict`: the code that runs it. It never raises: errors come back as data
  with a `status` (ok | insufficient_evidence | not_found | invalid_input | unavailable);
* `evidence_id` in every answer, so the agents can cite it.

`ToolContext` loads the team's files once (the two model tools from src/models, the processed
listings and reviews) and remembers why something could not be loaded, so a missing file
becomes an `unavailable` answer instead of a crash.
"""
from __future__ import annotations

import io
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pandas as pd

from agents.config import AgentConfig

LISTING_FILES = ["data/processed/listings_features.parquet",
                 "data/processed/listings_master.parquet",
                 "data/processed/listings_master.csv"]
REVIEW_FILES = ["data/processed/reviews_clean.parquet", "data/processed/reviews_amsterdam.parquet"]
# Only these columns are ever loaded: no host names, URLs, photos or free text except the title
DESCRIPTIVE_COLUMNS = [
    "location", "country", "id", "snapshot_date", "last_scraped", "name", "neighbourhood_cleansed",
    "latitude", "longitude",
    "property_type", "room_type", "bedrooms", "accommodates", "bathrooms_text", "price",
    "price_numeric", "minimum_nights", "availability_365", "number_of_reviews_ltm",
    "reviews_last_365d", "days_since_last_review", "review_scores_rating", "review_scores_location",
    "review_scores_cleanliness", "review_scores_value", "host_is_superhost",
    "calculated_host_listings_count", "license", "estimated_occupancy_l365d", "amenities_count",
    "has_wifi", "has_kitchen", "has_air_conditioning", "has_workspace", "has_parking",
    "comparable_competitors_1km"]


@dataclass
class ToolSpec:
    name: str
    schema: dict                                  # {"name", "description", "input_schema"}
    handler: Callable[[dict], dict] | None        # None for server tools (run by the API)
    read_only: bool = True
    server_tool: bool = False                     # executed by the Anthropic API (web search)
    external: bool = False                        # returns outside (web) content


def unavailable(tool: str, reason: str, **extra) -> dict:
    return {"tool": tool, "status": "unavailable", "status_reason": reason,
            "evidence_id": f"{tool}:unavailable", **extra}


@dataclass
class ToolContext:
    config: AgentConfig
    _cache: dict = field(default_factory=dict)
    errors: dict = field(default_factory=dict)

    @property
    def root(self) -> Path:
        return Path(self.config.data_root)

    def _load(self, key: str, loader):
        if key in self._cache:
            return self._cache[key]
        try:
            with warnings.catch_warnings(), redirect_stdout(io.StringIO()):
                warnings.simplefilter("ignore")
                value = loader()
            self.errors.pop(key, None)
        except Exception as e:                      # missing files, version mismatch, ...
            value = None
            self.errors[key] = f"{type(e).__name__}: {e}"
        self._cache[key] = value
        return value

    # ------------------------------------------------------------------ the two model tools
    def opportunity_tool(self):
        from src.models.opportunity_tool import OpportunityTool
        return self._load("opportunities", lambda: OpportunityTool.load(
            self.root, include_synthetic=True, verbose=False))

    def prediction_service(self):
        from src.models.predict import PredictionService

        def load():
            try:
                return PredictionService.load(self.root, verbose=False)
            except ValueError as e:
                # Older artifacts in models/ were trained on another listings file (e.g.
                # listings_master.parquet). The selected models use 03's feature file, so load
                # every city from it; PredictionTool still picks the model 05 selected.
                features = self.root / "data" / "processed" / "listings_features.parquet"
                if "different files" not in str(e) or not features.exists():
                    raise
                return PredictionService.load(self.root, data_file=features, verbose=False)
        return self._load("predictions", load)

    # ------------------------------------------------------------------ descriptive data
    def listings_file(self) -> Path | None:
        for rel in LISTING_FILES:
            if (self.root / rel).exists():
                return self.root / rel
        return None

    def reviews_file(self) -> Path | None:
        for rel in REVIEW_FILES:
            if (self.root / rel).exists():
                return self.root / rel
        return None

    def listings(self) -> pd.DataFrame | None:
        def load():
            path = self.listings_file()
            if path is None:
                raise FileNotFoundError("no processed listings file in data/processed/ "
                                        "(run notebooks 01-03 or unzip the data bundle)")
            if path.suffix == ".parquet":
                import pyarrow.parquet as pq
                cols = [c for c in DESCRIPTIVE_COLUMNS if c in pq.read_schema(path).names]
                df = pd.read_parquet(path, columns=cols)
            else:
                df = pd.read_csv(path, usecols=lambda c: c in DESCRIPTIVE_COLUMNS, low_memory=False)
            if "price_numeric" in df.columns:
                price = pd.to_numeric(df["price_numeric"], errors="coerce")
            else:
                price = pd.to_numeric(df["price"].astype("string").str.replace(r"[$,]", "", regex=True),
                                      errors="coerce")
            df["price_value"] = price.where(price > 0)
            if "location" not in df.columns:
                df["location"] = "unknown"
            # one spelling per city ("amsterdam" -> "Amsterdam"), matching the model and table names
            from src.models.cities import CITY_CURRENCY, city_slug
            canonical = {city_slug(c): c for c in CITY_CURRENCY}
            df["location"] = df["location"].astype(str).map(
                lambda c: canonical.get(city_slug(c), c.strip().title()) if c.strip() else "unknown")
            if "number_of_reviews_ltm" not in df.columns and "reviews_last_365d" in df.columns:
                df["number_of_reviews_ltm"] = df["reviews_last_365d"]
            if "snapshot_date" not in df.columns and "last_scraped" in df.columns:
                df["snapshot_date"] = df["last_scraped"]
            if "snapshot_date" in df.columns:          # one row per listing: its latest snapshot
                df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
                df = df.sort_values("snapshot_date").groupby(["location", "id"], as_index=False).tail(1)
            return df.reset_index(drop=True)
        return self._load("listings", load)

    def capabilities(self) -> dict[str, str | None]:
        """For each tool: None if its data is present, otherwise why it is unavailable. The
        orchestrator gives the agents only the tools that can answer, and tells them what is missing."""
        opp, pred, lst = self.opportunity_tool(), self.prediction_service(), self.listings()
        no_listings = None if lst is not None else "no processed listings in data/processed/"
        card = next(iter(Path(self.root, "reports").glob("*/model_card_metrics.json")), None)
        return {
            "rank_opportunities": None if opp else "no opportunity table in reports/<city>/ (run 06_experiments)",
            "predict_listing_price": None if pred else "no price model in models/ (run 04 and 05)",
            "model_card": None if card else "no model card in reports/<city>/ (run 05_model_evaluation)",
            "neighbourhood_stats": no_listings, "segment_profile": no_listings,
            "listing_profile": no_listings, "data_availability": None,
            "review_signals": None if self.reviews_file() else
                "guest review text is not in the repository (reviews_clean.parquet is too large for Git)",
            "roi_calculator": None, "web_search": None,
        }

    def status(self) -> dict:
        """What is loaded and what is missing (for the UI's side panel and data_availability)."""
        opp, pred, lst = self.opportunity_tool(), self.prediction_service(), self.listings()
        return {"data_root": str(self.root),
                "opportunity_tables": ({c: {"label": t.meta.get("label"),
                                            "snapshot_date": t.meta.get("snapshot_date"),
                                            "ranked": t.meta.get("ranked_segments"),
                                            "currency": t.meta.get("currency")}
                                        for c, t in opp.cities.items()} if opp else {}),
                "price_models": pred.cities if pred else [],
                "listings_file": str(self.listings_file()) if self.listings_file() else None,
                "listings_rows": int(len(lst)) if lst is not None else 0,
                "reviews_file": str(self.reviews_file()) if self.reviews_file() else None,
                "errors": dict(self.errors)}
