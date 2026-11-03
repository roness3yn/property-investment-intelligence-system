"""Agent evaluation harness (README criteria 7 and 8).

Runs the frozen investigation cases (22) in evaluation/eval_sets/agent_cases_v1.jsonl through the
full system (router, specialists, tools, guardrails) and reports, per case and overall:

* groundedness: share of the numbers in the answers that match the tool evidence log
  (criterion 7: at least 95%), and the share that is also cited;
* routing: whether the router chose the expected path;
* tools: whether the expected tools were used, and forbidden ones were not;
* missing-data behaviour: in every missing-data case a tool must have refused
  (insufficient_evidence / not_found / invalid_input / unavailable) and the answer must
  contain no ungrounded number;
* time, model calls and tokens per answer (the inputs to criterion 8's cost per brief).

    python -m agents.harness                      # all cases, writes evaluation/agent_results/
    python -m agents.harness --cases C01 C08      # a subset

It needs ANTHROPIC_API_KEY. Set PIIS_DATA_ROOT to the data bundle (or a demo repository).
The cases are frozen: change them only in a new file (agent_cases_v2.jsonl) and say so.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from agents.config import REPO_ROOT, AgentConfig
from agents.orchestrator import Orchestrator

CASES = REPO_ROOT / "evaluation" / "eval_sets" / "agent_cases_v1.jsonl"
REFUSALS = {"insufficient_evidence", "not_found", "invalid_input", "unavailable", "refused_by_policy"}


def tools_used(r) -> set[str]:
    out = {e["tool"] for e in r.events if e["type"] in ("tool_call", "tool_cached") and e.get("tool")}
    out |= {"ask_data_enrichment_agent" for e in r.events if e["type"] == "delegate"}
    return out


def score_case(case: dict, r) -> dict:
    v = r.verification or {}
    used = tools_used(r)
    statuses = {e.get("status") for e in r.events if e["type"] == "tool_result"}
    checks = {}
    expected = case.get("intent")
    checks["route"] = (r.mode == "blocked") if expected == "blocked" else (r.intent == expected)
    if case.get("tools_any"):
        checks["tools"] = bool(used & set(case["tools_any"]))
    if case.get("forbid_tools"):
        checks["no_forbidden_tools"] = not (used & set(case["forbid_tools"])) or \
            any(e.get("guard") == "policy" for e in r.events if e["type"] == "guardrail")
    if case.get("expect_insufficient"):
        checks["insufficient_evidence"] = bool(statuses & REFUSALS) and not v.get("ungrounded")
    if case.get("expect_blocked"):
        checks["blocked"] = r.mode == "blocked"
    if case.get("forbid_text"):
        checks["no_forbidden_text"] = not any(t.lower() in r.answer.lower() for t in case["forbid_text"])
    if v:
        checks["verified"] = bool(v.get("passed"))
    return {"id": case["id"], "kind": case.get("kind"), "question": case["question"],
            "intent": r.intent, "agent": r.agent, "mode": r.mode, "tools": sorted(used),
            "claims": v.get("claims", 0), "groundedness": v.get("groundedness"),
            "citation_rate": v.get("citation_rate"), "ungrounded": v.get("ungrounded", []),
            "violations": v.get("violations", []), "checks": checks, "passed": all(checks.values()),
            "seconds": r.elapsed_s, "llm_calls": r.usage["llm_calls"],
            "tokens": r.usage["input_tokens"] + r.usage["output_tokens"], "answer": r.answer}


def summarise(rows: list[dict]) -> dict:
    claims = sum(r["claims"] for r in rows)
    grounded = sum(round((r["groundedness"] or 0) * r["claims"]) for r in rows)
    cited = sum(round((r["citation_rate"] or 0) * r["claims"]) for r in rows)
    missing = [r for r in rows if "insufficient_evidence" in r["checks"]]
    briefs = [r for r in rows if r["mode"] == "agent"]
    return {"cases": len(rows), "cases_passed": sum(r["passed"] for r in rows),
            "numbers_checked": claims,
            "groundedness": grounded / claims if claims else None,
            "criterion_7_groundedness_met": (grounded / claims >= 0.95) if claims else None,
            "citation_rate": cited / claims if claims else None,
            "missing_data_cases": len(missing),
            "missing_data_correct": sum(r["checks"]["insufficient_evidence"] for r in missing),
            "criterion_7_missing_data_met": all(r["checks"]["insufficient_evidence"] for r in missing),
            "median_seconds_per_answer": statistics.median([r["seconds"] for r in briefs]) if briefs else None,
            "median_tokens_per_answer": statistics.median([r["tokens"] for r in briefs]) if briefs else None,
            "routing_accuracy": sum(r["checks"]["route"] for r in rows) / len(rows)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the frozen agent cases (criteria 7-8).")
    ap.add_argument("--cases", nargs="*", help="case ids, e.g. C01 C08")
    ap.add_argument("--out", default=str(REPO_ROOT / "evaluation" / "agent_results"))
    a = ap.parse_args()
    cases = [json.loads(l) for l in CASES.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.cases:
        cases = [c for c in cases if c["id"] in set(a.cases)]
    cfg = AgentConfig()
    if not cfg.api_key_present:
        raise SystemExit("set ANTHROPIC_API_KEY first (the harness evaluates the agents, not the fallback)")
    orch = Orchestrator(cfg)
    rows = []
    for c in cases:
        r = orch.handle(c["question"])
        row = score_case(c, r)
        rows.append(row)
        g = "-" if row["groundedness"] is None else f"{row['groundedness']:.0%}"
        print(f"{c['id']} {'PASS' if row['passed'] else 'FAIL'}  {row['agent']:<16} grounded {g:>4}  "
              f"{row['seconds']:>5.1f}s  {[k for k, ok in row['checks'].items() if not ok] or ''}")
    summary = summarise(rows)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M")
    meta = {"run": stamp, "cases_file": CASES.name, "agent_model": cfg.agent_model,
            "router_model": cfg.router_model, "data_root": str(cfg.data_root),
            "labels": {c: t["label"] for c, t in orch.ctx.status()["opportunity_tables"].items()}}
    (out / f"run_{stamp}.json").write_text(json.dumps({"meta": meta, "summary": summary, "cases": rows},
                                                      ensure_ascii=False, indent=2, default=str))
    print(json.dumps(summary, indent=2))
    print(f"written: {out / f'run_{stamp}.json'}")


if __name__ == "__main__":
    main()
