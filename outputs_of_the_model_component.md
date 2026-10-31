# Model outputs: what notebooks 04–06 hand to the agent and the interface

**Owner:** models & evaluation (Idan) · **Consumers:** agent (Adir), investor interface (Ron) · **Related:** [prediction_tool.md](prediction_tool.md), [opportunity_tool.md](opportunity_tool.md), README *Notebooks and data flow*

Notebooks 04–06 hand over two kinds of things:
- **files**: the trained models and the result tables;
- **two read-only Python tools** that read those files.

The agent uses the **tools**. The interface can use the **files** directly, the **tools**, or both.

## 1. What each notebook produces

| Notebook | Output | Location | Contents |
| --- | --- | --- | --- |
| `04_modeling` | `ridge_…`, `hist_gbm_…`, `hist_gbm_tuned_<city>_<version>.joblib`, each with a `.json` | `models/` | the trained price models; the metadata records features, settings, CV results, target policy and data fingerprint |
| | `cqr_<city>_<version>.joblib` | `models/` | the interval model that produces the 80% price ranges |
| | `04_g2_tuning_trials.csv` | `reports/<city>/` | the tuning trials (appendix material) |
| `05_model_evaluation` | `model_card_metrics.json` | `reports/<city>/` | **the selected model** (Amsterdam: `hist_gbm`) and its test metrics. The prediction tool reads this file to choose the model |
| | `04_*.csv` and `figures/*.png` | `reports/<city>/` | test results, segment tables, coverage, ablations and charts, for the report and slides; `04_readme_table.md` holds the README's results table |
| | `predictions_test_<city>_<split>.csv` | `models/` | per test listing: predicted price, 80% range, observed price and position (below / within / above) |
| | `test_runs.jsonl` | `evaluation/` | the official record of each test run (one-shot rule); **in Git** |
| `06_experiments` | `opportunities.csv` | `reports/<city>/` | **the ranked opportunities**: one row per neighbourhood × room type × bedrooms, with the three components, the score, the rank and the status |
| | `opportunity_listings.csv` | `reports/<city>/` | the evidence: one row per listing, with its opportunity, asking and expected price, 80% range and position |
| | `opportunities.json` | `reports/<city>/` | how the table was made: model, snapshot date, assumptions and label (`PROVISIONAL` or `FINAL`) |

## 2. The two tools

| Tool name | Code | Reads | Answers |
| --- | --- | --- | --- |
| `predict_listing_price` | `src/models/predict.py` (`PredictionService`) | the models and interval model from 04, the selection from 05, the processed listings | price, 80% range, position and drivers for **one listing**, existing (by id) or hypothetical (by features) |
| `rank_opportunities` | `src/models/opportunity_tool.py` (`OpportunityTool`) | the three files from 06 | `top`: the best-ranked opportunities; `breakdown`: one opportunity's components and score; `listings`: the listings behind one opportunity |

Both tools:
- are read-only, and always return the same keys;
- refuse rather than guess (`insufficient_evidence`);
- give an `evidence_id` to cite for every answer;
- never raise an exception at the caller.

Each tool exposes a `TOOL_SCHEMA` for the language model and a `handle_tool_call(args)` entry point. The details are in [prediction_tool.md](prediction_tool.md) and [opportunity_tool.md](opportunity_tool.md).

## 3. Who uses what

### Agent (Adir): the tools only

The agent never opens the files itself.
- `rank_opportunities`: *which opportunities should be investigated?* and *why does this one score high?*
- `predict_listing_price`: *is this listing priced below comparable listings?*
- Loading both tools with `evidence_log="evaluation/agent_evidence.jsonl"` logs every answer for the groundedness check (README criterion 7).

```python
from src.models.predict import PredictionService, TOOL_SCHEMA as PRICE_SCHEMA
from src.models.opportunity_tool import OpportunityTool, TOOL_SCHEMA as OPP_SCHEMA

log = "evaluation/agent_evidence.jsonl"
prices = PredictionService.load(".", evidence_log=log)
opps = OpportunityTool.load(".", evidence_log=log)

tools = [PRICE_SCHEMA, OPP_SCHEMA]                 # passed to the language model
handlers = {PRICE_SCHEMA["name"]: prices.handle_tool_call,
            OPP_SCHEMA["name"]: opps.handle_tool_call}
```

### Interface (Ron): files, tools, or both

- **Files:**
  - a ranked table from `opportunities.csv`;
  - a drill-down from `opportunity_listings.csv`;
  - a model-quality panel from `model_card_metrics.json`;
  - charts from `reports/<city>/figures/`.
- **Tools:** for example, a **what-if form** where an investor enters a property that is not listed yet (neighbourhood, room type, bedrooms, accommodates) and gets the expected nightly price and 80% range from `predict_listing_price`. The same call the agent uses:

```python
prices.handle_tool_call({"city": "Amsterdam",
                         "features": {"neighbourhood_cleansed": "De Pijp - Rivierenbuurt",
                                      "room_type": "Entire home/apt",
                                      "accommodates": 4, "bedrooms": 2}})
```

Whatever is shown must keep the guardrails: the score is conservative **rental revenue potential**, not a return on investment; booked nights are an approximation from reviews; nothing is a buy recommendation.

## 4. Getting the files: they are not in Git

`models/`, `reports/<city>/` and `data/processed/` are excluded from Git on purpose: they hold listing-level data and binary artifacts. The **code** (tools and docs) is in Git; the **files the tools read** are not.

| Needed | In Git? | How to get it |
| --- | --- | --- |
| `src/models/*.py`, `docs/*.md` | yes | `git pull` |
| `evaluation/feature_audit.csv`, `evaluation/splits/split_v1_<city>.json` | yes | `git pull` |
| `models/` (models, interval model, test predictions) | no | shared by the model owner, or produced by running 04 and 05 |
| `reports/<city>/` (model card, opportunity files, tables, figures) | no | shared by the model owner, or produced by running 05 and 06 |
| `data/processed/listings_features.parquet` | no | shared, or produced by running 01–03 on the same raw files |

To share: copy the three folders above into a zip, send it through a shared drive (not Git), and unzip it at the repository root, so the paths stay the same.

**Check that everything is in place,** from the repository root:

```
python -m src.models.predict --listing-id 28871
python -m src.models.opportunity_tool --city Amsterdam --top 5
```

The first prints a price prediction for an existing Amsterdam listing; the second prints the top five opportunities. An error naming a missing file shows which folder is missing.

## 5. Current status (Amsterdam, 30 Oct 2026)

| Output | Status |
| --- | --- |
| Models and interval model (04) | trained; the selected model is `hist_gbm` |
| Test evaluation (05) | run on 29 Oct 2026, labelled **exploratory** (see README *Evaluation*) |
| Opportunity table (06) | built; labelled `PROVISIONAL`, because 06 adds the test listings only after a confirmatory 05 run. Rerun 06 before sharing, so it uses the selected model |
