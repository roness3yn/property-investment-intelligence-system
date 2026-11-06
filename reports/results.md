# Results

Amsterdam, Inside Airbnb snapshot of 15 June 2026. Price-model results come from **one held-out test run on 29 Oct 2026, labelled exploratory**: the thresholds were fixed in `05_model_evaluation` before the run and nothing was changed after it, but the pre-registration commit was not recorded (see README, *Evaluation*). The test set is not scored again. The opportunity ranking was evaluated in `06_experiments` on 4 Nov 2026. The agent has not been evaluated yet.

## Baseline table

All rows use the same data and split: `listings_features.parquet`, split `split_v1_amsterdam.json` version `2a418f50c8dc` (grouped by listing id, no listing on both sides), scored on the **same 1,266 test listings**. Values in brackets are 95% cluster-bootstrap confidence intervals (2,000 resamples).

| Approach | Data/split | MAE (EUR/night) | RMSE (EUR/night) | Latency (median single call) | Cost |
| --- | --- | --- | --- | --- | --- |
| Global median (B0) | Amsterdam, `2a418f50c8dc`, 1,266 test listings | 132.6 [124.6, 140.6] | 197.6 [182.1, 213.4] | 0.4 ms | ≈ 0 |
| **Median baseline** (B1: training median by neighbourhood × room type, room-type and global fallback) | same | **112.0** [104.7, 119.4] | **174.1** [159.5, 188.5] | 1.9 ms | ≈ 0 |
| Linear model (ridge) | same | 79.2 [74.2, 84.6] | 123.9 [112.9, 134.5] | 7.9 ms | compute only (≈ 0) |
| **Learned price model** (gradient boosting `hist_gbm`, **selected**) | same | **75.0** [70.4, 79.7] | **113.0** [104.6, 121.3] | 9.5 ms; **27.7 ms** with the 80% range | compute only (≈ 0) |
| Tuned gradient boosting with kNN price feature (`hist_gbm_tuned`) | same | 74.9 [70.3, 79.7] | 113.7 [104.9, 122.4] | 14.5 ms | compute only (≈ 0) |
| Deterministic report (opportunity table, no agent) | Amsterdam, table `20261104_1753`, 5,049 listings, 46 ranked opportunities | N/A | N/A | Pending (agent harness) | ≈ 0 |
| Agent investigation | 22 frozen cases, `evaluation/eval_sets/agent_cases_v1.jsonl` | N/A | N/A | Pending (agent harness) | Pending (API cost) |

**Selection:** `hist_gbm`, by the pre-registered rule. The tuned model is not measurably better (MAE difference CI [−1.16, +1.33] EUR), so the simpler model is kept. Ridge is clearly worse (difference CI [+1.98, +6.82] EUR).

## Key results

### Price model (component 1)

| Measure | Model (`hist_gbm`) | Baseline (B1) |
| --- | --- | --- |
| MAE (EUR/night) | **75.0** | 112.0 (model **33.0% lower**; difference CI [−42.1, −32.1] EUR) |
| Typical relative error | 24.6% [23.4, 25.8] | 39.5% [37.4, 41.5] |
| Within ±20% of the actual price | 56.6% of listings [53.9, 59.4] | 38.5% [35.8, 41.1] |
| MAPE / median APE | 22.4% / 17.2% | 33.8% / 26.4% |
| Median absolute error (EUR/night) | 47.0 | 74.2 |
| R² on log price | 0.71 | 0.34 |
| Bias (EUR/night) | −19.1 [−25.5, −13.0] (under-predicts, mostly the most expensive listings) | −37.7 |

- **Consistent with development:** 5-fold grouped CV on the training partition gave 77.3 vs 112.8 EUR/night (31% lower, in all 5 folds).
- **No sign of leakage:** a model trained on shuffled prices does 4.9% worse than the global median, while real prices give 45% lower MAE.
- **By room type:** entire homes 84.2 vs 127.3 EUR/night (34% lower), private rooms 48.4 vs 67.5 (28% lower).
- **What drives accuracy (ablations, change in test MAE when a group is removed):** location features +10.7 EUR/night [8.1, 13.5], review features +2.8 [0.9, 4.8], amenity flags +0.8 (not significant), competitor counts −0.1 (not significant).
- **Availability features (deliberately excluded):** they would lower MAE by 3.8 EUR/night [1.6, 6.2], but they change which listings look over- or underpriced (residual rank correlation 0.893, below 0.90). Hypothesis H5 is supported, so they stay out.

### 80% price ranges (CQR)

| Measure | Result |
| --- | --- |
| Coverage | **80.8%** [78.7, 82.9] (target 77–83%) |
| Mean width | 242 EUR [236, 249] |
| Interval score | 369 EUR [342, 397] |
| Coverage, cheapest 10% / most expensive 10% of listings | 62.5% / 49.6% (below the 70% floor) |
| Coverage, entire homes / private rooms | 80.4% / 82.1% |

### Opportunity ranking (table `20261104_1753`, label PROVISIONAL, test listings not included)

- 124 opportunities (neighbourhood × room type × bedrooms); **46 ranked**, each with at least 20 priced listings, built from 5,049 listings.
- **Stability (criterion 5):** over 200 bootstrap resamples, the top 10 keeps **80%** of its opportunities on average (median rank correlation 0.94).
- **Robustness (criterion 6):** under the 8 alternative occupancy assumptions (review rate 30–70%, average stay 2–4 nights), the top 10 keeps **80–100%**, lowest at review rate 30% and 4-night stays.
- **Booked nights vs Inside Airbnb's own estimate** (`estimated_occupancy_l365d`): rank correlation 0.998 per listing and 0.87 per opportunity; median 24 vs 31 nights. This shows the approximation is implemented consistently, not that it equals real bookings.
- **Top two:**
  - **#1, Centrum-West, private room, 2 bedrooms:** 33,389 EUR/year (156 booked nights × 389 EUR × certainty 0.55). It rests on only 21 listings: in 44% of the resamples it falls below the 20-listing minimum and is not ranked.
  - **#2, Centrum-Oost, entire home, 2 bedrooms:** 22,436 EUR/year (73 × 493 × 0.62), in the top 10 in every resample.

### Success criteria

| # | Criterion | Result |
| --- | --- | --- |
| 1 | Price accuracy: MAE at least 10% below B1, CI of the difference excludes 0 | ✅ met (−33.0%) |
| 2 | Interval calibration: 77–83% coverage | ✅ met (80.8%) |
| 3 | Segment safety | ⚠️ partly met: no large segment worse than B1; coverage below 70% for the cheapest and most expensive 10% |
| 4 | Speed: under 200 ms per call | ✅ met (27.7 ms) |
| 5 | Ranking stability: top 10 keeps at least 70% | ✅ met (80%) |
| 6 | Ranking robustness: top 10 keeps at least 70% | ✅ met (lowest 80%) |
| 7 | Agent groundedness: at least 95% on 20 frozen cases | pending |
| 8 | Time saved: at least 25% median reduction per brief | pending |

### Hypotheses

| | Hypothesis | Result |
| --- | --- | --- |
| H1 | The learned model beats the baseline | supported (−33.0%) |
| H2 | Gains are largest for entire homes and dense central neighbourhoods | not supported (entire homes 33.8% vs others 29.3%; dense 32.9% vs others 33.7%) |
| H3 | 80% ranges reach 77–83% coverage | supported (80.8%) |
| H4 | Coverage drops on the next snapshot (drift) | not testable yet (one snapshot) |
| H5 | Availability lowers MAE but changes residual rankings | supported (−3.8 EUR; rank correlation 0.893) |

## Artifacts and reproducibility

| Item | Value |
| --- | --- |
| Data | `data/processed/listings_features.parquet`, Amsterdam rows SHA-256 `55802f265b1da8df34d41c60689758d9f2d2d500c49320b279522fd7dd609103` |
| Split | `evaluation/splits/split_v1_amsterdam.json`, version `2a418f50c8dc` (seed 42; test 20%, calibration 10%; stratified by room type × price quintile) |
| Feature schema | `evaluation/feature_audit.csv` (only `include` columns become features) |
| Selected model | `models/hist_gbm_amsterdam_20261024_1350.joblib` (+ `.json` metadata), interval model `models/cqr_amsterdam_20261024_1350.joblib` |
| Test run log | `evaluation/test_runs.jsonl` (29 Oct 2026, label EXPLORATORY) |
| Test predictions | `models/predictions_test_amsterdam_2a418f50c8dc.csv` |
| Metrics and tables | `reports/amsterdam/model_card_metrics.json`, `04_*.csv`, `05_relative_*.csv`, `06_ranking_checks.json`, `figures/` |
| Opportunity table | `reports/amsterdam/opportunities.csv`, `opportunity_listings.csv`, `opportunities.json` (version `20261104_1753`) |
| Agent evaluation cases | `evaluation/eval_sets/agent_cases_v1.jsonl`, cases C01–C22 |

**Pending (agent owner):** groundedness, failure counts, latency and cost of the deterministic report and the agent, and representative supported and unsupported claims, from `python -m agents.harness` on the frozen cases.
