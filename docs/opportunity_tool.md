# Opportunity tool: hand-off to the agent

**Owner:** models & evaluation (Idan) · **Consumer:** agent (Adir), investor view (Ron) · **Code:** `src/models/opportunity_tool.py`, scoring in `src/models/opportunities.py` · **Definition:** README, *Opportunity definition and score*


**Before first use:** run `06_experiments` for the city. It writes `reports/<city>/opportunities.csv`, `opportunity_listings.csv` and `opportunities.json`, which this tool reads. The files are local (not in Git), like the models.

## What is ranked

An **opportunity** is a combination of a neighbourhood, a room type and a number of bedrooms (0–1, 2, 3+). Its score, in local currency per year:

```
score = expected booked nights × expected nightly price × price certainty
```

| Component | How |
| --- | --- |
| Expected nightly price | median of the opportunity's listings' **out-of-fold** predictions (each listing predicted by a model that did not see its price) |
| Expected booked nights | mean over the listings of: reviews in the last 12 months ÷ 50% review rate × max(3 nights, minimum stay), capped at 70% of the year. Inactive listings count as 0. An approximation, not observed bookings |
| Price certainty | median of (lower end of the 80% price range ÷ expected price) |

Opportunities with fewer than 20 priced listings are `insufficient_evidence` and not ranked. The assumptions are in `opportunities.json` and in every answer (`assumptions`).

## Load once, call many times

```python
from src.models.opportunity_tool import OpportunityTool, TOOL_SCHEMA

tool = OpportunityTool.load(".")          # from the repository root; every city with a table
tool.handle_tool_call({"action": "top", "city": "Amsterdam", "n": 10})
```

Command line:
```
python -m src.models.opportunity_tool --city Amsterdam --top 10
python -m src.models.opportunity_tool --city Amsterdam --breakdown "De Pijp" "entire home" 2
python -m src.models.opportunity_tool --city Amsterdam --listings "De Pijp" "entire home" 2
```

## Actions

| `action` | Input | Returns |
| --- | --- | --- |
| `top` | optional `n` (default 10, max 50), `neighbourhood`, `room_type`, `bedrooms`, `include_insufficient` | the best-ranked opportunities, each with its components, score and a ready-made `statement` |
| `breakdown` | `neighbourhood`, `room_type`, `bedrooms` | one opportunity: the three components with how each is computed, and the `formula` with the numbers filled in |
| `listings` | `neighbourhood`, `room_type`, `bedrooms`, optional `n` (default 20) | the existing listings behind the opportunity, most below market first: asking price, out-of-fold expected price and 80% range, `position`, recent reviews |

Input is forgiving: `"De Pijp"` finds *De Pijp – Rivierenbuurt*, `"entire home"` finds *Entire home/apt*, bedrooms can be `2`, `"2"`, `"3+"` or `"studio"`. An ambiguous name returns `invalid_input` with the candidates.

## Output (always the same keys)

| Field | Meaning |
| --- | --- |
| `status` | `ok`, `insufficient_evidence`, `not_found` or `invalid_input` |
| `status_reason` | why, when `status` is not `ok` |
| `results` | list of opportunities (or listings, for `listings`) |
| `details` | action-specific extras (filters, counts), possibly empty |
| `currency` | the city's currency. Never compare across currencies |
| `label` | `FINAL` (after `05_model_evaluation`'s confirmatory run), `PROVISIONAL` (before it: test listings left out, development model) or `SYNTHETIC` |
| `assumptions` | the occupancy assumptions and the score formula |
| `guardrail` | what the numbers are not; put it in the agent's prompt too |
| `table_version`, `model_version`, `data_snapshot_date` | exactly which table, model and data answered |
| `evidence_id` | **cite this for every claim**, e.g. `opp:amsterdam:20261028_1203:breakdown:de_pijp_rivierenbuurt__entire_home_apt__2` |

## Guardrails (also for the agent's prompt)

1. No score or price is stated unless `status` is `ok`. For `insufficient_evidence`, the tool withholds the numbers.
2. The score is **conservative rental revenue potential**, not a return on investment, profit or property value: the data has no purchase prices or costs.
3. Booked nights are an **approximation from reviews**, not observed bookings.
4. Rank 1 is a **lead to investigate**, not a buy recommendation. The agent's job is to gather evidence for and against it.

## For the groundedness harness

```python
tool = OpportunityTool.load(".", evidence_log="evaluation/agent_evidence.jsonl")
```

Every call and answer is appended to the log, as with the prediction tool. A claim in the agent's brief is grounded if the `evidence_id` it cites exists in the log and the number it quotes matches the logged value.

## Tests

`python -m pytest tests/test_opportunities.py -q` (or `python -m tests.test_opportunities`). The tests cover:
- the bedroom bands;
- the occupancy formula;
- the score and the minimum-size rule;
- that out-of-fold predictions never use a listing's own price;
- the ranking checks;
- the tool's actions, its refusals and its fixed output keys.

No real data is needed.
