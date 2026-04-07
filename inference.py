"""
inference.py — OpenEnv baseline inference entrypoint.

Required by the OpenEnv validator at repo root.
Runs the AI agent across all three task levels (easy / medium / hard)
using the OpenAI API when OPENAI_API_KEY is set, otherwise falls back
to a deterministic oracle trajectory for offline validation.

Usage:
    python inference.py                          # mock run (no key needed)
    OPENAI_API_KEY=sk-... python inference.py    # live GPT-4o run
"""

import os
import json
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action

# ---------------------------------------------------------------------------
# OpenAI client — built once, shared across all tasks
# ---------------------------------------------------------------------------
def _build_client() -> OpenAI | None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[inference] No OPENAI_API_KEY found — using deterministic mock trajectory.")
        return None
    return OpenAI(api_key=api_key)


# ---------------------------------------------------------------------------
# Oracle trajectories (used when no API key is available)
# ---------------------------------------------------------------------------
_MOCK_ACTIONS: dict[str, list[Action]] = {
    "easy": [
        Action(action_type="Reply", content="Here is your password reset link: https://example.com/reset"),
        Action(action_type="CloseTicket"),
    ],
    "medium": [
        Action(action_type="SearchKB", content="warranty router defective"),
        Action(action_type="IssueRefund", amount=150.0),
        Action(action_type="Reply", content="I have issued a full refund for your defective router."),
        Action(action_type="CloseTicket"),
    ],
    "hard": [
        Action(action_type="SearchKB", content="billing issues double charge premium"),
        Action(action_type="Reply", content="I sincerely apologize for the double billing. I am processing a full refund immediately."),
        Action(action_type="IssueRefund", amount=1200.0),
        Action(action_type="CloseTicket"),
    ],
}

SYSTEM_PROMPT = """\
You are a customer support agent resolving tickets via structured tool calls.

Available actions — respond with EXACTLY one JSON object per turn:
  {"action_type": "Reply",       "content": "<message to customer>"}
  {"action_type": "SearchKB",    "content": "<search query>"}
  {"action_type": "IssueRefund", "amount": <float>}
  {"action_type": "Escalate"}
  {"action_type": "CloseTicket"}

Rules:
- Always SearchKB before issuing a refund.
- Apologize before closing a billing complaint.
- Close the ticket when fully resolved.
"""


def _llm_action(client: OpenAI, messages: list[dict]) -> Action:
    """Single OpenAI client call returning a validated Action."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    print(f"  LLM → {raw}")
    return Action(**json.loads(raw))


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------
def run_task(task_level: str, client: OpenAI | None) -> float:
    print(f"\n{'='*58}")
    print(f"  Task: {task_level.upper()}")
    print(f"{'='*58}")

    env = CustomerSupportEnv(task_level=task_level)
    obs = env.reset()

    # Mock path
    if client is None:
        print("  [MOCK] Running oracle trajectory.")
        for act in _MOCK_ACTIONS[task_level]:
            obs, reward, done, info = env.step(act)
            print(f"  {act.action_type:<15} | reward {reward:+.2f} | done={done}")
            if done:
                grade = info.get("grade", 0.0)
                print(f"  Grade: {grade:.2f}")
                return grade
        return 0.0

    # Live LLM path
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False
    while not done:
        messages.append({"role": "user", "content": f"Observation:\n{obs.model_dump_json(indent=2)}"})
        try:
            act = _llm_action(client, messages)
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"  [WARN] Bad LLM output ({exc}); defaulting to Reply.")
            act = Action(action_type="Reply", content="I am reviewing your issue now.")

        messages.append({"role": "assistant", "content": act.model_dump_json()})
        obs, reward, done, info = env.step(act)
        print(f"  {act.action_type:<15} | reward {reward:+.2f} | {info.get('feedback', '')}")

        if done:
            grade = info.get("grade", 0.0)
            print(f"  Grade: {grade:.2f}")
            return grade

    return 0.0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    client = _build_client()

    scores: dict[str, float] = {}
    for level in ["easy", "medium", "hard"]:
        scores[level] = run_task(level, client)

    print("\n=== INFERENCE RESULTS ===")
    for level, score in scores.items():
        print(f"  {level.capitalize():<8}: {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'Average':<8}: {avg:.2f}")
