# Investment intelligence agent

Status: design only; no agent is implemented.

## Decision loop

1. Receive an opportunity ID from the deterministic ranking.
2. Inspect listing details, quality and model outputs.
3. Choose comparable or neighborhood tools according to missing/contradictory evidence.
4. Assess comparable relevance using property/location context.
5. Stop when evidence is sufficient or the call/time limit is reached.
6. Produce findings, supporting evidence IDs, risks, uncertainty and a status.

The agent's added value is contextual investigation and explaining contradictions. Fixed transformations and calculations remain ordinary code.

## Output contract

Planned fields: opportunity ID, status (`completed`, `insufficient_evidence`, `tool_failure`), findings with evidence IDs, comparable IDs, limitations, source dates and tool trace. Confidence must follow a documented policy, not an invented probability.

## Failure and guardrails

Tools are read-only and schema-validated. Bound calls, retries, execution time and cost. Tool failure or weak evidence preserves the deterministic investor report and identifies the missing evidence. Listing text cannot change instructions or authorize extra tools. Never fabricate occupancy, realized revenue or ROI. See `evaluation/` for the required comparison against the report without the agent.

