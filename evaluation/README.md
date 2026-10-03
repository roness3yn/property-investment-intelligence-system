# Evaluation plan

Status: evaluation design; no measured results.

## Model evaluation

Freeze a one-city/currency holdout and record row counts, date ranges and split identifiers. Group repeated listing identities so they cannot cross train/test. Prefer temporal validation where snapshots permit it. Fit encoders, imputers and neighborhood baselines on training only; exclude target-derived features.

Compare a neighborhood/room-type median (global-median fallback) to the learned model using identical cases. Report MAE, RMSE, sample counts and errors by neighborhood, room type and price band. Include concrete failures and the limitations of nightly asking price as a target.

## Agent evaluation

Create 20 frozen cases covering complete data, missing data, contradictory evidence, unavailable-calendar proxies, nonexistent listing, tool timeout and prompt injection in listing text. For each case record expected evidence and response status.

Groundedness = supported factual claims / all factual claims; track no-claim responses separately. Human-label support against tool outputs and source snapshots. Compare the agent with the deterministic report on the same opportunities. Measure correctness, comparable relevance, missing-data behavior, latency, tool failures and cost. Keep tuning examples separate from frozen cases.

## Product evaluation

Run paired manual/system research tasks with the same goal and record participant/task counts, order, completion time and answer correctness. Report median time reduction and small-sample limitations. A/B testing is optional, not a substitute for the core evaluation.

Proposed numeric targets are in CHARTER.md; confirm before evaluation. Store cases in eval_sets/ and publish results and errors in reports/.

