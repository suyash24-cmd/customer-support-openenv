# Customer Support Resolution OpenEnv

A production-grade RL/Agent environment simulating real-world customer support ticket resolution. Designed to be compliant with the OpenEnv specification and deployable to Hugging Face Spaces.

## 1. Overview
Automating customer support requires agents to multi-task: navigate knowledge bases, manage distressed users, issue refunds correctly, and know when to escalate. This environment evaluates frontier LLMs on realistic, multi-step customer support scenarios to find failure modes in tool use and logic loops.

## 2. Environment Design

### State
The internal `State` includes the customer ticket, complete `CustomerContext` (premium status, spend, past tickets), the hidden `kb_articles` dictionary, full `conversation_history`, and metadata like `refund_issued` and `step_count`.

### Action Space
Structured via Pydantic model:
- `Reply(content: str)`: Talk to the user.
- `SearchKB(content: str)`: Search internal documentation.
- `IssueRefund(amount: float)`: Refund the customer.
- `Escalate()`: Send to human tier-2.
- `CloseTicket()`: End the conversation.

### Observation Space
Agents receive:
- `ticket`: Issue details.
- `customer_context`: Customer metadata.
- `conversation_history`: The live chat log.
- `last_action_feedback`: System feedback on the last action taken (e.g. results of a KB search).

## 3. Tasks
- **Easy**: A simple password reset request. Requires a reply with the proper link and ticket closure.
- **Medium**: A defective product report by a premium user. Requires searching the KB, verifying policy, issuing exactly a $150 refund, and closing.
- **Hard**: An angry customer double-billed for a subscription. Warrants apology, full refund, and possibly escalation. High ambiguity and risk of churn.

## 4. Reward Design
The environment provides a dense reward:
- `-0.05` per step to encourage efficiency.
- `-0.2` for redundant/invalid tool use.
- `+0.1` for KB searches (capped to prevent loops).
- Partial rewards for issuing correct intermediate refunds.
- Final task bonus up to `+10.0` based on a deterministic grader calculating percentage efficiency and correctness.

## 5. Setup Instructions

### Local Run
```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
$env:PYTHONPATH = "."; python scripts\run_baseline.py
```

### Docker Run
```bash
docker build -t openenv-support .
docker run -p 8000:8000 openenv-support
```

### Hugging Face Space
Upload this repository directly to a Hugging Face Docker Space. The `Dockerfile` exposes the FastAPI environment API on port 8000.

## 6. Baseline Results
- Easy: 0.8 to 1.0 depending on LLM parsing.
- Medium: 0.6 to 0.8
- Hard: 0.4 to 0.7
(Results visible after running `run_baseline.py`)
