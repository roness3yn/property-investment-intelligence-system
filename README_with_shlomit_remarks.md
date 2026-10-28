# Property Investment Intelligence System

BIU DS23 final project by **Erin Schaverien, Ron Ness, Idan Gilad and Adir Libovitch**.

> **Draft for team sign-off (28 Oct 2026).** This version answers the professor's two remarks: what the system ranks (**Opportunity definition and score**), and that the goals must be fixed before the results (**Data evidence** and **Success criteria**). Once the team agrees, remove this note and commit the README **together with the pre-registration commit, before `05_model_evaluation` is run on real data**. That commit is the record that the criteria came first.

## Purpose and status

Help short-term-rental investors decide **which type of property to investigate, and where**. The system ranks **opportunities**: combinations of a neighbourhood, a room type and a number of bedrooms. Each opportunity gets a score in currency per year, built from three defined components: the expected nightly price, the expected booked nights and the certainty of the price estimate. Agent investigations then gather the evidence for or against the top-ranked opportunities.

The product estimates **rental revenue potential**, not return on investment: the data has no purchase prices or operating costs (see Guardrails).

**Status: pipeline implemented, success criteria committed, test results pending.**
- Data preparation, EDA and feature engineering run on all cities in the Inside Airbnb extract.
- The nightly-price model, its evaluation and the agent's prediction tool are implemented per city (Amsterdam first).
- The opportunity score is implemented (`06_experiments`, `src/models/opportunities.py`), and its agent tool is `src/models/opportunity_tool.py` ([docs/opportunity_tool.md](docs/opportunity_tool.md)).
- The held-out test set has not been scored yet. Results will be judged against the **Success criteria** below, which were fixed before any test result existed.

## Data evidence (Amsterdam)

| | |
| --- | --- |
| City | Amsterdam, Netherlands (prices in EUR) |
| Snapshot | Inside Airbnb release of **15 June 2026** (`scrape_id` 20260615212022; listing pages scraped 16–24 June 2026) |
| Rows | **10,369 listings** (one row per listing, one snapshot) |
| Price coverage | **61.5%** of listings (6,377) have a non-empty, positive price. The other 38.5% are excluded from modelling and ranking, so results describe **bookable, priced listings** |
| License | Inside Airbnb data, Creative Commons Attribution 4.0 International (CC BY 4.0): free to use with attribution to Inside Airbnb |

**Data-quality gate (the first check).** A city is modelled only if **at least 50%** of its listings have a price **and at least 3,000 priced listings** remain. Price is the target, so its coverage is checked before anything else. Amsterdam passes (61.5%, 6,377 priced listings). Every city added later gets the same five-line block and the same gate.

## Success criteria (committed before results)

These thresholds are fixed **before** the test set is scored and the ranking is evaluated. A criterion that is not met is reported as **not met**; thresholds are not changed after the results are known. Criteria 1–4 are also coded in `05_model_evaluation`'s pre-registration (`PREREG`), so they are applied mechanically.

| # | Criterion | Committed threshold | Measured in | Result |
| --- | --- | --- | --- | --- |
| 1 | Price accuracy | test MAE **at least 10% below** the baseline (B1), **and** the 95% CI of the difference excludes 0 | `05_model_evaluation` (H1) | pending |
| 2 | Interval calibration | the 80% price range covers **77–83%** of test prices | `05_model_evaluation` (H3) | pending |
| 3 | Segment safety | no room type or neighbourhood with **at least 100** test listings is worse than B1; no segment's coverage is **below 70%** | `05_model_evaluation` | pending |
| 4 | Speed | prediction tool **under 200 ms** per call | `05_model_evaluation` | pending |
| 5 | Ranking stability | the top 10 keeps **at least 70%** of its opportunities under bootstrap resampling of the listings | `06_experiments` §7.3 | pending |
| 6 | Ranking robustness | the top 10 keeps **at least 70%** of its opportunities under the alternative occupancy assumptions (review rate 30–70%, average stay 2–4 nights) | `06_experiments` §7.4 | pending |
| 7 | Agent groundedness | **at least 95%** of the numbers the agent cites match the tool evidence log | agent harness (Adir) | pending |
| 8 | Time saved | the agent's evidence brief is **faster than the manual reference**, at no more than an agreed cost per brief | agent harness (Adir, Ron) | pending |

## Architecture

```mermaid
flowchart TD
    A["Inside Airbnb snapshots"] --> B["Validation and data pipeline"]
    B --> C["Feature engineering"]
    C --> D["Component 1: nightly-price model<br/>(expected price + 80% range)"]
    C --> O["Component 2: occupancy approximation<br/>(from review rate, stated assumptions)"]
    C --> E["Neighbourhood and comparable analytics"]
    D --> F["Opportunity score per neighbourhood x room type x bedrooms<br/>(price x booked nights x certainty) and quality checks"]
    O --> F
    F --> G["Investor interface"]
    F --> H["Agent investigation of top-ranked opportunities"]
    H --> I["Read-only analytical tools"]
    I --> D
    I --> E
    I --> F
    E --> H
    H --> G
```

The agent selects investigations, compares contextual evidence and explains contradictions. Cleaning, financial arithmetic, filtering, feature preparation, prediction and scoring remain in deterministic code. Investors can inspect the ranked table when the agent is unavailable.

## Opportunity definition and score

**An opportunity is a combination of a neighbourhood, a room type and a number of bedrooms, and its score is calculated from three defined components: the expected nightly price, the expected booked nights per year and the certainty of the price estimate.**

### Unit

| Element | Definition |
| --- | --- |
| Neighbourhood | `neighbourhood_cleansed` (Inside Airbnb's official neighbourhoods) |
| Room type | `room_type`: entire home/apt, private room, shared room, hotel room |
| Bedrooms | bands: 0–1, 2, 3+ |
| Eligibility | at least **20 priced listings** in the city's snapshot; smaller combinations are listed as "insufficient evidence", not ranked |

Example: *De Pijp – Rivierenbuurt × entire home/apt × 2 bedrooms.* This is the kind of choice an investor makes before buying: what to buy and where. Existing listings are evidence about each opportunity, not the things being ranked, since they are not for sale.

### The three components

| # | Component | Question it answers | How it is computed |
| --- | --- | --- | --- |
| 1 | **Expected nightly price** (EUR/night) | How much will such a property earn per booked night? | The nightly-price model's prediction for the opportunity's listings: the median of their **out-of-fold** predictions (each listing is predicted by a model that did not see it), so a listing's own price never sets its own estimate |
| 2 | **Expected booked nights** (nights/year) | How often will it be booked? | **Stated approximation from the review rate**, the method Inside Airbnb uses for `estimated_occupancy_l365d`: booked nights = reviews in the last 12 months ÷ review rate × average stay, where review rate = **50%** of stays get a review, average stay = **3 nights** or the listing's minimum stay if longer, capped at **70%** of the year (255 nights). The opportunity's value is the **mean** over its listings (inactive listings count as 0) |
| 3 | **Price certainty** (0–1) | How reliable is component 1? | Lower end of the 80% price range ÷ expected price, from the conformal (CQR) intervals. 1 = a precise estimate; lower = more spread among comparable properties |

### Score

```
Opportunity score (EUR/year) = expected booked nights × expected nightly price × price certainty
                             ≈ expected booked nights × lower end of the 80% price range
```

The score is a **conservative annual rental revenue**: the revenue at the pessimistic end of the normal price range. Uncertain opportunities are discounted automatically, in the same unit as the revenue. The output table also shows the **expected annual revenue** (components 1 × 2, without the discount) and each component separately, so an investor can see why an opportunity ranks where it does.

Output: `reports/<city>/opportunities.csv`, one row per opportunity, with:
- the three components, the score and the rank;
- the number of listings;
- the share of listings with a review in the last 12 months;
- an evidence identifier.

The agent reads it through a read-only tool, `src/models/opportunity_tool.py`: top-N with filters, the score breakdown of one opportunity, and the listings behind it ([docs/opportunity_tool.md](docs/opportunity_tool.md)).

### Occupancy decision

Occupancy is **inside the score, as a stated approximation**. Reviews are only a proxy for bookings: not every guest reviews, and review rates differ by segment. The assumptions above are therefore stated, not estimated, and criterion 6 tests how much the ranking depends on them.

*Considered and not chosen:* keeping occupancy outside the ranking (ranking on price and certainty only). It avoids the proxy, but the score would then be per night, and could not answer how much a property earns per year.

## Decisions

| Decision | Chosen | Alternative considered |
| --- | --- | --- |
| Bedroom bands | 0–1 / 2 / 3+ | exact bedroom count (more, smaller groups) |
| Minimum size of an opportunity | 20 priced listings | a lower threshold (more opportunities, noisier) |
| Occupancy | inside the score, stated approximation | outside the ranking, as context |
| Third component | price certainty | competition (listings within 1 km of the same room type) |
| Inactive listings | 0 booked nights (conservative) | left out of the average |
| Plan B | a second Inside Airbnb city | Zillow Research (different source and target) |

## Team

| Responsibility | Owner |
| --- | --- |
| Data and features, data evidence block | Erin Schaverien |
| Models, quantitative evaluation and the opportunity score | Idan Gilad |
| Agent, tools, guardrails and the investigation-task harness | Adir Libovitch |
| Product, architecture and investor output | Ron Ness |

Everyone contributes code and presents.

## Repository map

| Path | Contents |
| --- | --- |
| `CHARTER.md` | Business problem, scope, metrics, risks and milestones |
| `data/raw/`, `data/processed/` | Local datasets, excluded from Git |
| `data/data_dictionary.md` | Planned schema; replace with inspected source schema |
| `data/sources.md` | Source provenance per city: URL, download date, snapshot, rows, price coverage, license, checksum |
| `notebooks/` | Ordered EDA, preprocessing, modelling, evaluation and experiments |
| `src/` | Reusable ingestion, features, models, analytics and application code |
| `agents/` | Decision loop, tools, prompts and guardrails |
| `evaluation/` | Frozen evaluation cases, groundedness and error analysis |
| `models/` | Local model artifacts and version metadata |
| `reports/` | Baseline comparisons, results, figures and the opportunity table |
| `presentation/` | Final presentation and demo backup |
| `docs/` | Architecture decisions and delivery plan |
| `tests/` | Tests for the prediction-tool contract, city handling, engineered features and the opportunity score (`python -m pytest tests`) |

## Notebooks and data flow

Run the notebooks in order. Each reads the previous step's output from `data/processed/` (not in Git).

| Notebook | Owner | Reads | Writes |
| --- | --- | --- | --- |
| `01_Data_Preparation` | Erin | raw CSVs in `data/listings/`, `data/reviews/` | `listings_master.parquet`, `reviews_clean.parquet` |
| `02_EDA` | Erin | the files above | charts and maps in `reports/` |
| `03_Feature_Engineering` | Erin | the files above | `listings_features.parquet` (listings + engineered columns) |
| `04_modeling` | Idan | `listings_features.parquet`, `evaluation/splits/split_v1_<city>.json` | models in `models/` (not in Git) |
| `05_model_evaluation` | Idan | the same data and split, the saved models | `reports/<city>/`, `evaluation/test_runs.jsonl` |
| `06_experiments` | Idan | `listings_features.parquet`, the selected model and interval model, the split | `reports/<city>/opportunities.csv`, `opportunity_listings.csv`, `opportunities.json`; the ranking checks |

Until `05_model_evaluation` has made its one confirmatory test run, `06_experiments` leaves the test listings out of the opportunity table and labels it `PROVISIONAL`.

Raw files are named `<city>_listings.csv` and `<city>_reviews.csv`, with the same lowercase city name in both (e.g. `amsterdam_listings.csv`).

**One model per city, one currency per model.** `04_modeling`, `05_model_evaluation` and `06_experiments` take a `CITY` parameter. Each city needs its own frozen split, created once from the repository root:

```bash
python -m src.models.make_split --data data/processed/listings_features.parquet --city Amsterdam
```

Which columns may become model features is decided in `evaluation/feature_audit.csv`. Price-derived columns are excluded as target leakage. `05_model_evaluation` scores each city's test set **once**: later runs are labelled exploratory.

The agent gets predictions through a read-only tool, `src/models/predict.py`, described in [docs/prediction_tool.md](docs/prediction_tool.md).

The *Modelling & Evaluation Design* document numbers its notebook blueprints differently: its "03_modeling" is `04_modeling` here, and its "04_model_evaluation" is `05_model_evaluation`.

## Environment and working order

```bash
git clone https://github.com/roness3yn/property-investment-intelligence-system.git
cd property-investment-intelligence-system
python -m venv .venv
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

`requirements.txt` pins the libraries the notebooks and `src/` use. In VS Code, select the project's `.venv` as the kernel of **every** notebook; packages installed elsewhere are not visible to it. Shared code lives in `src/` (`src/models/` for modelling, evaluation and the prediction tool). The dashboard and LLM provider are chosen later.

## Data and reproducibility

Source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), city by city. The verified facts for each modelled city are in its **data evidence** block (above, for Amsterdam) and in `data/sources.md`, together with the source URL, download date and file checksum.

**Plan B: a second Inside Airbnb city.** If Amsterdam fails the data-quality gate, or a second city is needed, the next city comes from the same source. It has the same columns and runs through the same pipeline by changing `CITY`. It is chosen by the same gate (price coverage at least 50%, at least 3,000 priced listings). Zillow Research, the charter's original fallback, is no longer the plan: it has a different source, country and target (property values, not nightly prices), so it would need a new pipeline.

Do not commit large datasets, credentials, host personal details or trained binary artifacts. Use local ignored directories or controlled external storage; document reproducible download steps. Record transformations and missing-value handling in the dictionary. Keep duplicate listing identities across snapshots out of both sides of the evaluation split, and fit preprocessing only on training data.

## Evaluation

The product is evaluated at three levels, each tied to the **Success criteria** above. **MAE is a quality check on component 1, not the product's goal:** an accurate nightly price does not by itself tell an investor where to buy.

### 1. Component quality (criteria 1–4)

| Component | Measure | Status |
| --- | --- | --- |
| 1. Expected nightly price | MAE (EUR/night) vs the baseline on held-out listings (criterion 1) | development result below; test pending |
| 1/3. 80% price range | coverage (criterion 2: 77–83%), width, interval score | calibration coverage 80.3%; test pending |
| 2. Booked nights | agreement with Inside Airbnb's `estimated_occupancy_l365d`; sensitivity of the ranking to the assumptions (criterion 6) | implemented in `06_experiments` |

All alternatives use the same held-out cases and split.

| Approach | MAE in selected city's currency/night | RMSE | Response time | Cost/query |
| --- | --- | --- | --- | --- |
| Training-set median by neighbourhood and room type, room-type and global fallback | Not measured | Not measured | Not measured | Not measured |
| Learned nightly-price model | Not measured | Not measured | Not measured | Not measured |
| Deterministic opportunity table without agent | N/A | N/A | Not measured | Not measured |
| Agent investigation of the same top-ranked opportunities | N/A | N/A | Not measured | Not measured |

*Development result (Amsterdam, June 2026 snapshot, 5-fold grouped CV on the training partition only):* the baseline's MAE is 112.8 EUR/night and gradient boosting's is 77.3 EUR/night (31% lower, in all 5 folds). A shuffled-price check shows no sign of leakage. Tuning and a kNN price feature gave no further gain. These are not test results and do not count toward criterion 1.

### 2. Ranking quality (criteria 5–6)

- **Stability (criterion 5):** resample the listings (bootstrap, 200 times). The top 10 must keep at least 70% of its opportunities on average. The rank correlation is reported too.
- **Robustness (criterion 6):** rerun the score with review rates of 30%, 50% and 70% and average stays of 2, 3 and 4 nights. The top 10 must keep at least 70% of its opportunities.
- **Forward check** (needs a later snapshot of the same city, not a criterion yet): did the top-ranked opportunities keep higher prices and review activity than the rest?

### 3. Investigation task (criteria 7–8)

**Task:** produce an evidence brief for each of the top 10 opportunities in a city: the score breakdown, comparable listings, review signals and contradictions, with every number cited.

The task is done three ways: by the agent, by the deterministic table alone, and manually by a team member as the reference. They are compared on time per brief, cost per query, groundedness (cited numbers that match the tool logs, criterion 7) and failed tool calls.

### Metrics for component 1

The model predicts a number (the nightly listing price), so it is evaluated with regression metrics, all on the same held-out listings:

- **MAE** in the city's currency per night is the primary metric. It is easy to interpret and robust to the few very expensive listings.
- **RMSE** shows whether there are large misses. **MedAE**, **typical relative error** (a percentage error based on log ratios) and **hit rate within ±20%** describe typical and relative accuracy. **R²** is computed on log price.
- MAPE is not used, because it explodes on cheap listings.
- Every metric is reported with a 95% cluster-bootstrap confidence interval, overall and per segment (room type, neighbourhood, price decile).

**Target transformation.** Prices are right-skewed, so models train on log(1 + price) and convert back with exp. This estimates the *median* listing price, which is what MAE rewards. RMSE, which rewards the mean, uses Duan's smearing correction from out-of-fold residuals. Interval bounds are quantiles and need no correction. Details: `notebooks/05_model_evaluation.ipynb`.

**Tuning.** Whether tuning pays off depends on the city. The pipeline tests it per city and keeps the simpler model unless the added complexity measurably helps.

The success criteria above are binding. `CHARTER.md` and `evaluation/README.md` must state the same thresholds.

## Guardrails

- The score is **conservative annual rental revenue potential** for a type of property in a place. It is not return on investment, profit or property value: the data has no purchase prices or operating costs.
- Compute ROI only with explicit acquisition and operating-cost inputs supplied by the user.
- Booked nights are an **approximation from reviews** with stated assumptions, not observed bookings. Availability is not confirmed occupancy.
- Nightly asking price is not realized revenue: guests may pay less after discounts.
- Results describe **priced listings** only (61.5% of Amsterdam's listings).
- An opportunity with fewer than 20 priced listings, or whose price estimate the tool refuses, is reported as insufficient evidence and not ranked.
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
