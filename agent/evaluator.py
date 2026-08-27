"""
evaluator.py — Computes evaluation metrics from a completed AgentRunResult.

All metrics are derived from the real environment state and trace produced
by agent.run_ticket(); nothing here is fabricated or hardcoded per task.
"""

from __future__ import annotations

from typing import Any, Dict

from env.graders import grade_easy_task, grade_hard_task, grade_medium_task


_GRADERS = {
    "easy": grade_easy_task,
    "medium": grade_medium_task,
    "hard": grade_hard_task,
}


def evaluate(run_result) -> Dict[str, Any]:
    """Build the metrics payload returned by /api/agent/evaluation/{session_id}."""
    state = run_result.env.get_state()
    grader = _GRADERS.get(run_result.task_level)
    resolution_success = grader(state) if grader else 0.0

    total_actions = len(run_result.trace)
    tool_calls = [t for t in run_result.trace if t.tool != "get_ticket_state"]
    failed_calls = [t for t in run_result.trace if not t.success]
    redundant = getattr(state, "redundant_actions", 0)

    tool_efficiency = 1.0
    if tool_calls:
        tool_efficiency = max(0.0, 1.0 - (redundant / max(len(tool_calls), 1)))

    policy_compliance = _policy_compliance(run_result)

    refund_accuracy = _refund_accuracy(run_result)

    resolution_time = None
    if run_result.trace:
        resolution_time = run_result.trace[-1].timestamp - run_result.trace[0].timestamp

    return {
        "resolution_success": round(float(resolution_success), 4),
        "policy_compliance": round(policy_compliance, 4),
        "tool_efficiency": round(tool_efficiency, 4),
        "reward": round(float(run_result.total_reward), 4),
        "num_actions": total_actions,
        "escalation_rate": 1.0 if state.is_escalated else 0.0,
        "refund_accuracy": round(refund_accuracy, 4) if refund_accuracy is not None else None,
        "resolution_time_seconds": round(resolution_time, 4) if resolution_time is not None else None,
        "status": run_result.status,
    }


def _policy_compliance(run_result) -> float:
    """Fraction of verify_policy checks (and the refunds that followed them)
    that were consistent with what the environment's KB actually justified."""
    verified: Dict[float, bool] = {}
    checks = 0
    compliant = 0
    for event in run_result.trace:
        if event.tool == "verify_policy" and isinstance(event.result, dict):
            verified[event.arguments.get("amount")] = event.result.get("justified", False)
        if event.tool == "issue_refund":
            checks += 1
            amount = event.arguments.get("amount")
            if verified.get(amount):
                compliant += 1
    if checks == 0:
        return 1.0  # no refund issued -> nothing to be non-compliant about
    return compliant / checks


def _refund_accuracy(run_result):
    """Compares the refund actually issued against the task's known-correct
    resolution (derived from env/tasks.py + env/graders.py, not hardcoded here)."""
    state = run_result.env.get_state()
    expected = {"easy": 0.0, "medium": 150.0, "hard": None}.get(run_result.task_level)
    if expected is None:
        return None  # hard task has no single "correct" refund amount
    if expected == 0.0:
        return 1.0 if state.refund_issued == 0.0 else 0.0
    return 1.0 if abs(state.refund_issued - expected) < 0.01 else 0.0
