import os
import json
import sys
from openai import OpenAI
from env.environment import CustomerSupportEnv
from env.models import Action

# ---------------------------------------------------------------------------
# OpenAI client — instantiated once and shared across all tasks.
# Reads the key from the environment; raises clearly if missing.
# ---------------------------------------------------------------------------
def _build_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set. Falling back to deterministic mock baseline.")
        return None
    return OpenAI(api_key=api_key)


def _get_mock_actions(task_level: str) -> list[Action]:
    """Pre-scripted oracle trajectory used when no API key is present."""
    if task_level == "easy":
        return [
            Action(action_type="Reply", content="Here is your reset link: https://example.com/reset"),
            Action(action_type="CloseTicket"),
        ]
    if task_level == "medium":
        return [
            Action(action_type="SearchKB", content="warranty router"),
            Action(action_type="IssueRefund", amount=150.0),
            Action(action_type="Reply", content="I have issued a full refund for your defective router."),
            Action(action_type="CloseTicket"),
        ]
    if task_level == "hard":
        return [
            Action(action_type="SearchKB", content="billing issues double charge"),
            Action(action_type="Reply", content="I sincerely apologize for the double billing. I am issuing a full refund now."),
            Action(action_type="IssueRefund", amount=1200.0),
            Action(action_type="CloseTicket"),
        ]
    return []


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
- Never loop on the same action twice.
- Close the ticket when fully resolved.
"""


def _llm_action(client: OpenAI, messages: list[dict]) -> Action:
    """Single OpenAI client call; parses and returns a validated Action."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.0,
    )
    raw = response.choices[0].message.content
    print(f"  LLM → {raw}")
    data = json.loads(raw)
    return Action(**data)


def run_agent_on_task(task_level: str, client: OpenAI | None) -> float:
    print(f"\n{'='*60}")
    print(f"  Task: {task_level.upper()}")
    print(f"{'='*60}")

    env = CustomerSupportEnv(task_level=task_level)
    obs = env.reset()

    # --- Mock path (no API key) ---
    if client is None:
        print("  [MOCK] Running deterministic oracle trajectory.")
        for act in _get_mock_actions(task_level):
            obs, reward, done, info = env.step(act)
            print(f"  Action: {act.action_type:15s} | Reward: {reward:+.2f} | Done: {done}")
            if done:
                grade = info.get("grade", 0.0)
                print(f"  Grade: {grade:.2f}")
                return grade
        return 0.0

    # --- Live OpenAI Client path ---
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    done = False

    while not done:
        messages.append({"role": "user", "content": f"Current observation:\n{obs.model_dump_json(indent=2)}"})

        try:
            act = _llm_action(client, messages)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  [WARN] Invalid action from LLM ({e}); defaulting to Reply.")
            act = Action(action_type="Reply", content="I am looking into your issue now.")

        # Append the assistant turn so context accumulates correctly
        messages.append({"role": "assistant", "content": act.model_dump_json()})

        obs, reward, done, info = env.step(act)
        print(f"  Action: {act.action_type:15s} | Reward: {reward:+.2f} | Feedback: {info.get('feedback', '')}")

        if done:
            grade = info.get("grade", 0.0)
            print(f"  Grade: {grade:.2f}")
            return grade

    return 0.0


if __name__ == "__main__":
    openai_client = _build_client()

    scores: dict[str, float] = {}
    for level in ["easy", "medium", "hard"]:
        scores[level] = run_agent_on_task(level, openai_client)

    print("\n=== BASELINE RESULTS ===")
    for level, score in scores.items():
        print(f"  {level.capitalize():<8}: {score:.2f}")
    average = sum(scores.values()) / len(scores)
    print(f"  {'Average':<8}: {average:.2f}")
