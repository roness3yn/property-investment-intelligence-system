# Property Investment Intelligence System

BIU DS23 final project by **Erin Schaverien, Ron Ness, Idan Gilad and Adir Libovitch**.

## Purpose and status

Help short-term-rental investors identify listings and neighborhoods worth further investigation through predictive modelling, market analytics and evidence-grounded agent investigations.

**Status: pipeline implemented, final results pending.** Data preparation, EDA and feature engineering run on all cities in the Inside Airbnb extract. Modelling, evaluation and the agent's prediction tool are implemented per city (Amsterdam first). The held-out test set has not been scored yet, so the evaluation table below has no results.

The uploaded Project Charter defines Inside Airbnb as the primary source and Zillow Research as Plan B. The working document also describes an earlier Zillow-led market scope. This repository follows the charter's Airbnb scope. Acquisition value and ROI cannot be inferred from nightly listing prices.

## Architecture

```mermaid
flowchart TD
    A["Inside Airbnb snapshots"] --> B["Validation and data pipeline"]
    B --> C["Feature engineering"]
    C --> D["Nightly-price prediction"]
    C --> E["Neighborhood and comparable analytics"]
    D --> F["Opportunity ranking and quality checks"]
    E --> F
    F --> G["Investor interface"]
    F --> H["Agent investigation"]
    H --> I["Read-only analytical tools"]
    I --> D
    I --> E
    H --> G
```

The agent selects investigations, compares contextual evidence and explains contradictions. Cleaning, financial arithmetic, filtering, feature preparation, prediction and ranking remain in deterministic code. Investors can inspect analytical outputs when the agent is unavailable.

## Team

| Responsibility | Owner |
| --- | --- |
| Data and features | Erin Schaverien |
| Models and quantitative evaluation | Idan Gilad |
| Agent, tools and guardrails | Adir Libovitch |
| Product, architecture and investor output | Ron Ness |

Everyone contributes code and presents.

## Repository map

| Path | Contents |
| --- | --- |
| `CHARTER.md` | Business problem, scope, proposed metrics, risks and milestones |
| `data/raw/`, `data/processed/` | Local datasets, excluded from Git |
| `data/data_dictionary.md` | Planned schema; replace with inspected source schema |
| `data/sources.md` | Source provenance and acquisition checklist |
| `notebooks/` | Ordered EDA, preprocessing, modelling, evaluation and experiments |
| `src/` | Reusable ingestion, features, models, analytics and application code |
| `agents/` | Decision loop, tools, prompts and guardrails |
| `evaluation/` | Frozen evaluation cases, groundedness and error analysis |
| `models/` | Local model artifacts and version metadata |
| `reports/` | Baseline comparisons, results and figures |
| `presentation/` | Final presentation and demo backup |
| `docs/` | Architecture decisions and delivery plan |
| `tests/` | Tests for the prediction-tool contract, city handling and engineered features (`python -m pytest tests`) |

## Notebooks and data flow

Run the notebooks in order. Each reads the previous step's output from `data/processed/` (not in Git).

| Notebook | Owner | Reads | Writes |
| --- | --- | --- | --- |
| `01_Data_Preparation` | Erin | raw CSVs in `data/listings/`, `data/reviews/` | `listings_master.parquet`, `reviews_clean.parquet` |
| `02_EDA` | Erin | the files above | charts and maps in `reports/` |
| `03_Feature_Engineering` | Erin | the files above | `listings_features.parquet` (listings + engineered columns) |
| `04_modeling` | Idan | `listings_features.parquet`, `evaluation/splits/split_v1_<city>.json` | models in `models/` (not in Git) |
| `05_model_evaluation` | Idan | the same data and split, the saved models | `reports/<city>/`, `evaluation/test_runs.jsonl` |
| `06_experiments` | Idan | the same data and split, the saved models | |

**One model per city, one currency per model.** `04_modeling` and `05_model_evaluation` take a `CITY` parameter. Each city needs its own frozen split, created once from the repository root:

```bash
python -m src.models.make_split --data data/processed/listings_features.parquet --city Amsterdam
```

Which columns may become model features is decided in `evaluation/feature_audit.csv`. Price-derived columns are excluded as target leakage. `05_model_evaluation` scores each city's test set **once**: later runs are labelled exploratory.

The agent gets predictions through a read-only tool, `src/models/predict.py`, described in [docs/prediction_tool.md](docs/prediction_tool.md).

The *Modelling & Evaluation Design* document numbers its notebook blueprints differently: its "03_modeling" is `04_modeling` here, and its "04_model_evaluation" is `05_model_evaluation`.

## Environment and working order

Use Python 3.11 as the proposed team baseline. The environment and dependencies must be verified before the first pipeline checkpoint.

```bash
git clone https://github.com/roness3yn/property-investment-intelligence-system.git
cd property-investment-intelligence-system
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1`.

`requirements.txt` intentionally contains no dependencies yet: library versions must be chosen and pinned when implementation starts. Installing it does not make the application runnable. Run the notebooks in the order above. Shared code lives in `src/` (`src/models/` for modelling, evaluation and the prediction tool). Choose the dashboard and LLM provider later.

## Data and reproducibility

Primary source: [Inside Airbnb](https://insideairbnb.com/get-the-data/).
Fallback: [Zillow Research](https://www.zillow.com/research/data/).

The charter reports approximately 1.5 million listings across 153 CSV files and roughly 12 months of coverage. These are planning figures, not counts verified by this repository. Record city, source URL, download date, snapshot date, row counts, units, license and checksum in `data/sources.md`. Begin with one city and one currency.

Do not commit large datasets, credentials, host personal details or trained binary artifacts. Use local ignored directories or controlled external storage; document reproducible download steps. Record transformations and missing-value handling in the dictionary. Keep duplicate listing identities across snapshots out of both sides of the evaluation split, and fit preprocessing only on training data.

## Evaluation and cheaper baseline

All alternatives must use the same held-out cases and split. Values are pending implementation.

Amsterdam MAE's:

| Approach | MAE in selected city's currency/night | RMSE | Response time | Cost/query |
| --- | --- | --- | --- | --- |
| Training-set median by neighborhood and room type, global-median fallback | Not measured | Not measured | Not measured | Not measured |
| Learned nightly-price model | Not measured | Not measured | Not measured | Not measured |
| Deterministic analytical report without agent | N/A | N/A | Not measured | Not measured |
| Agent investigation of the same opportunities | N/A | N/A | Not measured | Not measured |

**Metrics.** The model predicts a number (the nightly listing price), so it is evaluated with regression metrics, all on the same held-out listings:

- **MAE** in the city's currency per night is the primary metric (charter target: at least 10% below the baseline). It is easy to interpret and robust to the few very expensive listings.
- **RMSE** shows whether there are large misses. **MedAE**, **typical relative error** (a percentage error based on log ratios) and **hit rate within ±20%** describe typical and relative accuracy. **R²** is computed on log price.
- MAPE is not used, because it explodes on cheap listings.
- Every metric is reported with a 95% cluster-bootstrap confidence interval, overall and per segment (room type, neighbourhood, price decile).
- The 80% price interval is evaluated by **coverage** (target 77–83%), **width** and **interval score**.

**Target transformation.** Prices are right-skewed, so models train on log(1 + price) and convert back with exp. This estimates the *median* listing price, which is what MAE rewards. RMSE, which rewards the mean, uses Duan's smearing correction from out-of-fold residuals. Interval bounds are quantiles and need no correction. Details: `notebooks/05_model_evaluation.ipynb` .

**Tunning** Whether tuning pays off depends on the city; our pipeline tests it per city and keeps the simpler model unless the added complexity measurably helps.

Proposed targets and the evaluation design are in `CHARTER.md` and `evaluation/README.md`. Report model error, groundedness, failed tool calls, uncertainty, and whether the extra agent cost produces useful evidence.

## Guardrails

- Availability is not confirmed occupancy or bookings.
- Nightly asking price is not acquisition value or realized revenue.
- Compute ROI only with explicit acquisition and operating-cost inputs.
- Ground claims in tool outputs with source dates and evidence identifiers.
- Return insufficient evidence when data is missing or contradictory.
- The product supports investigation; it does not issue buy/no-buy instructions.

## Course deadlines

Charter: **4 October 2026**. Recommended pipeline checkpoint: **18 October**.
Recommended agent checkpoint: **30 October**. Pre-mentoring repository freeze: **1 November**.
Mentoring: **4 November**. Final slides and repository freeze: **12 November**.
Presentations: **15 or 18 November**, subject to team allocation.

See [CHARTER.md](CHARTER.md) and [delivery plan](docs/delivery-plan.md).

## Contributions

Work in short feature branches, open a pull request and describe the behavior and verification. Use issues to track ownership and acceptance criteria. See [CONTRIBUTING.md](CONTRIBUTING.md).

