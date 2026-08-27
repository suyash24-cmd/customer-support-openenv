"""
inference.py — OpenEnv baseline inference entrypoint.

Required by the OpenEnv validator at repo root.
Runs the Gemini + Google ADK agent (agent.agent.run_ticket) across all three
task levels (easy / medium / hard). Uses the live Gemini API when
GOOGLE_API_KEY (or Vertex AI credentials) is configured, otherwise falls
back to a deterministic oracle trajectory for offline validation — no
network access or API key required to run this file.

Usage:
    python inference.py                          # mock run (no key needed)
    GOOGLE_API_KEY=... python inference.py       # live Gemini run
"""

from dotenv import load_dotenv

from agent.agent import run_ticket
from agent.evaluator import evaluate

load_dotenv()


def main() -> dict[str, float]:
    scores: dict[str, float] = {}
    for level in ["easy", "medium", "hard"]:
        print(f"\n{'=' * 58}")
        print(f"  Task: {level.upper()}")
        print(f"{'=' * 58}")

        result = run_ticket(session_id=f"inference-{level}", task_level=level)
        for event in result.trace:
            print(f"  {event.tool:<22} | reward {event.reward:+.2f}")

        metrics = evaluate(result)
        grade = metrics["resolution_success"]
        scores[level] = grade
        print(f"  Status: {result.status} | Grade: {grade:.2f} | Reward: {metrics['reward']:.2f}")

    print("\n=== INFERENCE RESULTS ===")
    for level, score in scores.items():
        print(f"  {level.capitalize():<8}: {score:.2f}")
    avg = sum(scores.values()) / len(scores)
    print(f"  {'Average':<8}: {avg:.2f}")
    return scores


if __name__ == "__main__":
    main()
