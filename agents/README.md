# Investment intelligence agents

Status: **implemented** (Router, Opportunity Investigator, Price Analyst, Regulation & Compliance,
Data Enrichment Agent, Evidence Reviser), with three guardrail layers, a deterministic regulation
check and an investor chat. Full description:
[docs/agents.md](../docs/agents.md).

| Path | Contents |
| --- | --- |
| `orchestrator.py` | one question in, one checked answer out: input guardrail → router → specialist → output guardrail → reviser; deterministic fallback |
| `llm.py` | the agent loop (Anthropic SDK), agent-as-a-tool delegation, web-search pause/resume |
| `tools/` | read-only tools: the two model tools from `src/models`, plus neighbourhood statistics, review signals, data availability, ROI calculator, regulation rules, compliance check, web search |
| `regulation/` | Amsterdam's short-term-rental rules (official sources, dates) and the 15-night wijk boundaries; `compliance.py` applies them per listing |
| `guardrails/` | input (injection, size), tools (allow-list, schema, budget, policy, timeout), output (groundedness of every number, citations, policy) |
| `prompts/` | versioned system prompts |
| `evidence.py` | the evidence store: references E1, E2, … and the log for the groundedness harness |
| `harness.py` | runs the 20 frozen cases in `evaluation/eval_sets/agent_cases_v1.jsonl` (criteria 7–8) |
| `demo_data.py` | builds a clearly labelled synthetic demo repository when the real data bundle is absent |
| `cli.py` | ask from the terminal |

## Decision loop (the Investigator)

1. `rank_opportunities` top or breakdown: what ranks where, and why.
2. `listings` for the opportunity, then `review_signals` on its listings: what guests say.
3. `neighbourhood_stats`: how the neighbourhood compares with the city.
4. Missing data or outside context → the Data Enrichment Agent, called as a tool.
5. Stop when the evidence is sufficient or the tool budget is reached.
6. Brief: bottom line, the score and its components, evidence for, evidence against,
   what to check next; every number cited [E#], and checked before it is shown.

## Output contract

`TurnResult` (`orchestrator.py`): the answer; the intent and agent; the trace of events (agents,
tool calls, guardrail decisions, verification); the evidence records (reference, tool, arguments,
status, `evidence_id`, result); the verification report (groundedness, citation rate, ungrounded
numbers, policy violations); usage (model calls, tokens) and time; and the mode (`agent`,
`fallback`, `blocked`, `canned`).
