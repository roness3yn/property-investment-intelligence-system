# Project Charter — Property Investment Intelligence System

**Team:** Erin Schaverien (Data), Idan Gilad (Model), Adir Libovitch (Agent), Ron Ness (Product).
**Charter submission:** 4 October 2026. **Revised:** 30 October 2026, after the professor's remarks: the ranked unit and score are defined, the success criteria are committed with numbers, the data is verified, and Plan B is a second Inside Airbnb city. The [README](README.md) holds the full definitions and results; this charter states the commitments, and the two must agree.

## Problem and user

Short-term-rental investors spend time manually comparing neighbourhoods, property types, nightly prices, activity indicators and competitors before deciding what to buy and where. The system **ranks opportunities** and explains why the top-ranked ones warrant investigation.

An **opportunity** is a combination of a neighbourhood, a room type and a number of bedrooms (0–1, 2, 3+), with at least 20 priced listings. Its score is a **conservative annual rental revenue**:

```
score (local currency/year) = expected booked nights × expected nightly price × price certainty
```

The product estimates rental revenue potential, **not return on investment**: the data has no purchase prices or operating costs. Research-time savings are a hypothesis to measure (criterion 8), not a demonstrated outcome.

## Success criteria (committed before results)

These thresholds are fixed before the corresponding results exist and are not changed afterwards. A criterion that is not met is reported as not met. The README's **Success criteria** table records the results.

| # | Criterion | Committed threshold | Status (30 Oct 2026) |
| --- | --- | --- | --- |
| 1 | Price accuracy | held-out MAE **at least 10% below** the baseline (training-only median by neighbourhood × room type, with room-type and global fallback), and the 95% CI of the difference excludes 0 | met: −33.0% (exploratory run) |
| 2 | Interval calibration | the 80% price range covers **77–83%** of held-out prices | met: 80.8% |
| 3 | Segment safety | no room type or neighbourhood with at least 100 held-out listings is worse than the baseline; no segment's coverage is below 70% | partly met: coverage below 70% for the cheapest and most expensive 10% of listings |
| 4 | Speed | prediction tool under **200 ms** per call | met: 27.7 ms |
| 5 | Ranking stability | the top 10 keeps **at least 70%** of its opportunities under bootstrap resampling of the listings | pending |
| 6 | Ranking robustness | the top 10 keeps **at least 70%** of its opportunities under alternative occupancy assumptions (review rate 30–70%, average stay 2–4 nights) | pending |
| 7 | Agent groundedness | **at least 95%** of the numbers the agent cites match the tool evidence log, on **20 frozen investigation cases**, with correct insufficient-evidence behaviour in every missing-data case | pending |
| 8 | Time saved | **at least 25% median reduction** in time per evidence brief versus the manual reference, at no more than an agreed cost per brief | pending |

Criteria 1–4 are coded in `05_model_evaluation`'s pre-registration, so they are applied mechanically. The first held-out run (Amsterdam, 29 Oct 2026) is labelled **exploratory**: the thresholds were fixed before it and nothing was changed after it, but the pre-registration commit was not recorded. The test set is not scored again; a confirmatory test needs a later snapshot. Small samples in criteria 7–8 are reported as exploratory, with participant and task counts.

## Data and Plan B

Primary source: [Inside Airbnb](https://insideairbnb.com/get-the-data/), city by city. Start with one city and one currency; one model per city.

**Data evidence (Amsterdam), inspected:**

| | |
| --- | --- |
| City | Amsterdam, Netherlands (EUR) |
| Snapshot | Inside Airbnb release of 15 June 2026 |
| Rows | 10,369 listings |
| Price coverage | 61.5% (6,377 listings with a positive price) |
| License | Creative Commons Attribution 4.0 (CC BY 4.0) |

**Data-quality gate:** a city is modelled only if at least 50% of its listings have a price and at least 3,000 priced listings remain. Price coverage is the first check, because price is the target. Amsterdam passes.

**Plan B: a second Inside Airbnb city**, chosen by the same gate. It has the same schema and runs through the same pipeline. Zillow Research, the original fallback, is dropped: a different source, country and target (property values, not nightly prices) would need a new pipeline.

## Architecture and the agent's seat

Inside Airbnb → validation → features → nightly-price model (expected price + 80% range) and occupancy approximation → opportunity score and ranking → investor output, with an agent investigating the top-ranked opportunities over read-only tools. See the [architecture diagram](README.md#architecture).

Code performs cleaning, filtering, calculations, prediction and scoring. The agent chooses which opportunities and comparables to examine, weighs contextual signals, investigates contradictions and explains the evidence. Its tools are read-only: `predict_listing_price` (price, range and drivers for one listing) and `rank_opportunities` (top opportunities, score breakdowns, the listings behind each), plus listing details, neighbourhood statistics and data-quality checks.

Agent failure returns the deterministic opportunity table and an explicit insufficient-evidence or tool-failure status. No fabricated values or unsupported financial calculations.

## Risks and cut scope

| Risk | Response |
| --- | --- |
| Missing price data | Price coverage checked first (the data-quality gate); results describe priced listings only |
| Missing/stale data | Validate fields and snapshot dates; flag low-quality cases |
| Leakage across snapshots | Separate listing identities; fit preprocessing on training only; price-derived features excluded by the feature audit |
| Occupancy mistaken for bookings | Occupancy is a **stated approximation** from review rates, never presented as observed bookings; criterion 6 tests how much the ranking depends on its assumptions |
| Misleading financial signals | No ROI without cost inputs; the score is rental revenue potential, not property value |
| Unreliable ranges at the price extremes | Reported as a limitation (criterion 3); flags are least reliable for the cheapest and most expensive listings |
| Results tuned after seeing the test set | One-shot rule: each test set is scored once and logged; later runs are labelled exploratory |
| Unsupported agent claims | Evidence identifiers, bounded tool calls and insufficient-evidence responses |

If only two weeks remain, deliver one-city ingestion, a dictionary, the median baseline, one model, the opportunity table and investor output. Preserve this path without an agent. The course requires an agentic component for the final deliverable, so an agent-free cut is a recovery stage; integrate a narrow opportunity-investigation agent before the final freeze. Defer additional cities, extra models and optional A/B testing.

## Milestones and ownership

| Deliverable | Date | Owner | Status (30 Oct 2026) |
| --- | --- | --- | --- |
| Data and features, original team target | 30 Sep 2026 | Erin | Amsterdam pipeline runs (notebooks 01–03) |
| Charter and inspected-data evidence | 4 Oct 2026 | Team / Erin | charter submitted; data evidence added 28 Oct |
| Modelling, original team target | 7 Oct 2026 | Idan | done: models trained (04), held-out evaluation run 29 Oct (05) |
| Agent integration, original team target | 14 Oct 2026 | Adir | *owner to update*; tools ready for integration |
| End-to-end baseline checkpoint | 18 Oct 2026 | Team | *team to update* |
| Product/output, original team target | 24 Oct 2026 | Ron | *owner to update* |
| Opportunity score and ranking checks | 28 Oct 2026 | Idan | implemented (06); real-data ranking checks pending |
| Agent checkpoint and one-minute demo | 30 Oct 2026 | Adir / Ron | *owner to update* |
| Pre-mentoring repository freeze | 1 Nov 2026 | Team | |
| Internal final-version target from charter | 2 Nov 2026 | Team | |
| Mentoring with working system and draft slides | 4 Nov 2026 | Team | |
| Final repository and slides freeze | 12 Nov 2026 | Team | |
| Presentation, allocated session | 15 or 18 Nov 2026 | Team | |

Course freeze dates take precedence over the internal 2 November target.

## Checkpoint updates

At each checkpoint record: what runs end to end; metric versus target; decisions and reasons; blockers; the next three steps.

**Checkpoint, 30 Oct 2026**
- **Runs end to end (Amsterdam):** data preparation, features, price model, held-out evaluation, opportunity table, and both agent tools.
- **Metrics versus target:** criteria 1, 2 and 4 met; criterion 3 partly met; criteria 5–8 pending (see the table above).
- **Decisions:** the opportunity unit and score; occupancy inside the score as a stated approximation; Plan B a second Inside Airbnb city; default gradient boosting selected over the tuned model (no measurable gain).
- **Blockers:** the held-out run is exploratory (the pre-registration commit was missed); the agent harness for criteria 7–8 is not built yet.
- **Next three steps:** run the ranking checks on real data (criteria 5–6); integrate the tools into the agent and build the harness (criteria 7–8); fix the cost-per-brief limit for criterion 8.

**Source basis:** uploaded DS23 Final Project Guidelines (sections 3–6), Property Investment Intelligence System Project Charter (sections 1–6), team row 11 of the charter workbook, the professor's remarks of October 2026, and the Airbnb-only architecture image.
