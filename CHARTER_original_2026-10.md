# Project Charter — Property Investment Intelligence System

**Team:** Erin Schaverien (Data), Idan Gilad (Model), Adir Libovitch (Agent), Ron Ness (Product).
**Charter submission:** 4 October 2026. **Status:** initial draft derived from uploaded charter; proposed metrics below require team confirmation.

## Problem and user

Short-term-rental investors spend time manually comparing listings, neighborhoods, nightly prices, activity indicators and competitors. The system narrows the research set and explains why selected opportunities warrant investigation. Research-time savings are a hypothesis to measure, not a demonstrated outcome.

## Success criteria and cheaper baseline

Proposed primary target: reduce held-out nightly-price MAE by **at least 10%** versus a training-only neighborhood/room-type median with global-median fallback. Report RMSE and segment errors on the same split. Select one city/currency initially; listing price is the modelling target, not acquisition value.

Proposed agent target: **at least 90%** of factual claims supported by recorded tool evidence on **20 frozen investigation cases**, with correct insufficient-evidence behavior in every missing-data case. Compare to a deterministic report using the same opportunities; record response time, tool failures and cost. Proposed product target: **at least 25% median reduction** in research-task completion time in paired manual/system tasks. Record participant and task counts; small samples are exploratory. Confirm these targets before evaluating.

## Data and Plan B

Primary: [Inside Airbnb](https://insideairbnb.com/get-the-data/), including listings, calendar and review data where available. The uploaded charter reports approximately 1.5 million listings across 153 CSV files and approximately 12 months of coverage. Counts, schemas and access are not verified in this scaffold; Data owner must record an inspected city sample, dates, row counts and license before submission. Start with one city, one currency and the smallest defensible feature set.

Fallback: [Zillow Research](https://www.zillow.com/research/data/). If Airbnb access or usable price coverage fails, change to U.S. market-level housing/rental analysis and revise the target, data dictionary and evaluation. These sources are not interchangeable.

## Architecture and the agent's seat

Inside Airbnb → validation → features → price prediction and market analytics → opportunity ranking → investor output. An optional investigation path uses an agent over read-only analytical tools. See the [architecture diagram](README.md#architecture).

Code performs cleaning, filtering, calculations, scoring and prediction. The agent chooses relevant comparables, weighs contextual signals, investigates contradictions and explains evidence. Tools: listing details, model predictions, comparable search, neighborhood statistics, competition/activity indicators and data-quality checks.

Agent failure returns the deterministic report and an explicit insufficient-evidence or tool-failure status. No fabricated values or unsupported financial calculations.

## Risks and cut scope

| Risk | Response |
| --- | --- |
| Missing/stale data | Validate fields and snapshot dates; flag low-quality cases |
| Leakage across snapshots | Separate listing identities; use temporal splits when feasible; fit preprocessing on training only |
| Availability mistaken for bookings | Label unavailable-day proxies explicitly; make no occupancy inference |
| Misleading financial signals | No ROI without cost inputs; distinguish nightly price from property value |
| Unsupported agent claims | Evidence identifiers, bounded tool calls and insufficient-evidence responses |

If only two weeks remain, deliver one-city ingestion, a dictionary, a median baseline, one model, market analytics and investor output. Preserve this path without an agent. The course requires an agentic component for the final deliverable, so an agent-free cut is a recovery stage; integrate a narrow comparable-investigation agent before the final freeze. Defer additional cities, extra models and optional A/B testing.

## Milestones and ownership

| Deliverable | Date | Owner |
| --- | --- | --- |
| Data and features, original team target | 30 Sep 2026 | Erin |
| Charter and inspected-data evidence | 4 Oct 2026 | Team / Erin |
| Modelling, original team target | 7 Oct 2026 | Idan |
| Agent integration, original team target | 14 Oct 2026 | Adir |
| End-to-end baseline checkpoint | 18 Oct 2026 | Team |
| Product/output, original team target | 24 Oct 2026 | Ron |
| Agent checkpoint and one-minute demo | 30 Oct 2026 | Adir / Ron |
| Pre-mentoring repository freeze | 1 Nov 2026 | Team |
| Internal final-version target from charter | 2 Nov 2026 | Team |
| Mentoring with working system and draft slides | 4 Nov 2026 | Team |
| Final repository and slides freeze | 12 Nov 2026 | Team |
| Presentation, allocated session | 15 or 18 Nov 2026 | Team |

The 30 September target has passed; completion is unverified. Preserve it as the original target and record actual status in the next checkpoint. Course freeze dates take precedence over the internal 2 November target.

## Checkpoint updates

At each checkpoint record: what runs end to end; metric versus target; decisions and reasons; blockers; the next three steps. No checkpoint results recorded yet.

**Source basis:** uploaded DS23 Final Project Guidelines (sections 3–6), Property Investment Intelligence System Project Charter (sections 1–6), team row 11 of the charter workbook, and Airbnb-only architecture image. The older working document's Zillow-led scope is retained only as Plan B.

