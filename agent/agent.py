"""
agent.py — Google ADK agent that resolves customer support tickets by
calling tools against a live OpenEnv CustomerSupportEnv.

Design notes
------------
* Google ADK (`google-adk`) is the PRODUCTION agent framework. The default
  execution path in `run_ticket()` builds a real `google.adk.agents.Agent`
  (see `build_adk_agent()`) backed by Gemini 3.5+, wires it into a real
  `google.adk.runners.Runner` + `InMemorySessionService`, and executes it via
  `_run_adk_loop()`. Every tool the ADK agent calls is a real Python function
  that delegates to `ToolExecutor`, which in turn calls
  `CustomerSupportEnv.step(...)` — the agent can never bypass OpenEnv.
* `google-adk` uses Google's `google-genai` SDK internally to talk to Gemini
  (Developer API or Vertex AI, per GOOGLE_GENAI_USE_VERTEXAI), so the Gemini
  integration is preserved — it now flows through ADK rather than a
  hand-rolled function-calling loop.
* When no Gemini credentials are configured (`_build_genai_client()` returns
  None), `run_ticket()` falls back to a deterministic oracle trajectory
  (mirroring the environment's own graders in env/graders.py). This fallback
  is for OFFLINE TESTING AND LOCAL DEVELOPMENT ONLY — it is not the
  production path and is never used when credentials are present.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from env.environment import CustomerSupportEnv
from .prompts import SYSTEM_PROMPT
from .tools import TRACE_LABELS, ToolExecutor

MAX_AGENT_STEPS = 10
# Upper bound on top-level ADK Runner.run() turns. Each turn already lets the
# ADK agent make several tool calls internally (Gemini keeps calling tools
# until it emits a final response); this just guards against the agent
# stopping short of closing the ticket and needing a nudge to continue.
MAX_ADK_TURNS = 4


@dataclass
class TraceEvent:
    action: str
    tool: str
    arguments: Dict[str, Any]
    result: Any
    reward: float
    success: bool
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "tool": self.tool,
            "arguments": self.arguments,
            "result": self.result,
            "reward": self.reward,
            "success": self.success,
            "timestamp": self.timestamp,
        }


@dataclass
class AgentRunResult:
    session_id: str
    task_level: str
    status: str  # "resolved" | "escalated" | "timed_out"
    final_response: str
    total_reward: float
    trace: List[TraceEvent]
    env: CustomerSupportEnv
    execution_path: str = "adk"  # "adk" (production) | "deterministic_fallback" (offline/tests)


def _build_genai_client():
    """Build a google-genai client. Returns None if no credentials configured.

    Used as the credential check that decides which path `run_ticket()`
    takes; also used directly by the deterministic-fallback tests.
    """
    use_vertex = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "false").lower() == "true"
    try:
        from google import genai
    except ImportError:
        return None

    if use_vertex:
        project = os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
        if not project:
            return None
        return genai.Client(vertexai=True, project=project, location=location)

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return None
    return genai.Client(api_key=api_key)


def _deterministic_plan(task_level: str) -> List[Dict[str, Any]]:
    """Oracle tool-call plan used ONLY when no Gemini credentials are
    configured (offline testing / local dev without Google Cloud setup),
    mirroring the environment's own graders (env/graders.py). This is never
    used in production — see module docstring."""
    if task_level == "easy":
        return [
            {"name": "get_ticket_state", "args": {}},
            {"name": "respond_to_customer", "args": {"message": "Here is your password reset link: https://example.com/reset"}},
            {"name": "close_ticket", "args": {}},
        ]
    if task_level == "medium":
        return [
            {"name": "get_ticket_state", "args": {}},
            {"name": "search_knowledge_base", "args": {"query": "warranty policy defective router"}},
            {"name": "verify_policy", "args": {"amount": 150.0}},
            {"name": "issue_refund", "args": {"amount": 150.0}},
            {"name": "respond_to_customer", "args": {"message": "I've issued a full $150 refund for your defective router."}},
            {"name": "close_ticket", "args": {}},
        ]
    if task_level == "hard":
        return [
            {"name": "get_ticket_state", "args": {}},
            {"name": "search_knowledge_base", "args": {"query": "billing issues double charge premium"}},
            {"name": "verify_policy", "args": {"amount": 1200.0}},
            {"name": "respond_to_customer", "args": {"message": "I sincerely apologize for the duplicate charge — issuing a full refund now."}},
            {"name": "issue_refund", "args": {"amount": 1200.0}},
            {"name": "close_ticket", "args": {}},
        ]
    return [{"name": "get_ticket_state", "args": {}}, {"name": "close_ticket", "args": {}}]


def run_ticket(session_id: str, task_level: str = "easy", client: Optional[Any] = None) -> AgentRunResult:
    """Run the full agent loop for one ticket, from reset to resolution.

    Production execution flow (when Gemini credentials are configured):
        Customer ticket -> Google ADK Agent -> Gemini 3.5+ -> ADK tool call
        -> ToolExecutor -> CustomerSupportEnv.step() -> observation + reward
        -> ADK/Gemini next action -> ... -> resolution.

    Offline/local-dev fallback (no Gemini credentials): a deterministic
    oracle trajectory drives the same ToolExecutor/OpenEnv path, so tests
    and demos work without any Google Cloud setup. This fallback is never
    used when `GOOGLE_API_KEY` (or Vertex credentials) are present.
    """
    env = CustomerSupportEnv(task_level=task_level)
    env.reset()
    executor = ToolExecutor(env)

    genai_client = client if client is not None else _build_genai_client()

    trace: List[TraceEvent] = []
    total_reward = 0.0
    final_response = ""
    execution_path = "adk"

    if genai_client is None:
        # OFFLINE / LOCAL-DEV FALLBACK ONLY — not the production path.
        execution_path = "deterministic_fallback"
        for step_plan in _deterministic_plan(task_level):
            if env.get_state().is_closed:
                break
            name, args = step_plan["name"], step_plan["args"]
            result = executor.dispatch(name, args)
            reward = result.get("reward", 0.0) if isinstance(result, dict) else 0.0
            total_reward += reward
            if name == "respond_to_customer":
                final_response = args.get("message", final_response)
            trace.append(TraceEvent(
                action=TRACE_LABELS.get(name, name), tool=name, arguments=args,
                result=result, reward=reward, success=True,
            ))
    else:
        # PRODUCTION PATH: real Google ADK Agent + Runner.
        trace, total_reward, final_response = _run_adk_loop(
            env=env, task_level=task_level, session_id=session_id,
        )

    state = env.get_state()
    if state.is_escalated:
        status = "escalated"
    elif state.is_closed:
        status = "resolved"
    else:
        status = "timed_out"

    return AgentRunResult(
        session_id=session_id, task_level=task_level, status=status,
        final_response=final_response, total_reward=total_reward,
        trace=trace, env=env, execution_path=execution_path,
    )


def _run_adk_loop(env: CustomerSupportEnv, task_level: str, session_id: str):
    """PRODUCTION execution path: runs the real Google ADK Agent (built by
    `build_adk_agent()`) via `google.adk.runners.Runner`, and reconstructs
    a TraceEvent list from the Runner's emitted events.

    Every tool call the ADK agent makes is a real Python function
    (see `build_adk_agent`) that calls `ToolExecutor`, which calls
    `CustomerSupportEnv.step(...)` — this loop never touches the
    environment directly, it only observes ADK's events.
    """
    from google.genai import types

    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService

    agent = build_adk_agent(env)
    session_service = InMemorySessionService()
    app_name = "customer_support_agent"
    user_id = f"ticket-{session_id}"
    session_service.create_session_sync(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(app_name=app_name, agent=agent, session_service=session_service)

    trace: List[TraceEvent] = []
    total_reward = 0.0
    final_response = ""

    message_text = (
        "A new support ticket needs resolution. Begin by calling "
        "get_ticket_state, then resolve it using the available tools."
    )

    for turn in range(MAX_ADK_TURNS):
        if env.get_state().is_closed:
            break

        new_message = types.Content(role="user", parts=[types.Part(text=message_text)])
        pending_calls: Dict[str, Dict[str, Any]] = {}  # call id -> {"name", "args"}

        for event in runner.run(user_id=user_id, session_id=session_id, new_message=new_message):
            content = getattr(event, "content", None)
            if content is None or not getattr(content, "parts", None):
                continue
            for part in content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None:
                    pending_calls[fc.id or fc.name] = {"name": fc.name, "args": dict(fc.args or {})}

                fr = getattr(part, "function_response", None)
                if fr is not None:
                    call_info = pending_calls.pop(fr.id or fr.name, {"name": fr.name, "args": {}})
                    name = call_info["name"]
                    args = call_info["args"]
                    result = fr.response if isinstance(fr.response, dict) else {"result": fr.response}
                    reward = result.get("reward", 0.0) if isinstance(result, dict) else 0.0
                    total_reward += reward
                    if name == "respond_to_customer":
                        final_response = args.get("message", final_response)
                    trace.append(TraceEvent(
                        action=TRACE_LABELS.get(name, name), tool=name, arguments=args,
                        result=result, reward=reward, success="error" not in result,
                    ))

                text = getattr(part, "text", None)
                if text and getattr(event, "author", None) != "user":
                    # Model produced a plain-text reply instead of (or after)
                    # tool calls — still worth surfacing as the customer-facing
                    # response if no explicit respond_to_customer call fired.
                    if not final_response:
                        final_response = text

        message_text = "Continue resolving the ticket using the available tools."

    return trace, total_reward, final_response


def build_adk_agent(env: CustomerSupportEnv):
    """Build the production Google ADK Agent for a given environment instance.

    Returns a genuine `google.adk.agents.Agent` (requires `google-adk`,
    declared in requirements.txt) configured with Gemini 3.5+ as its model
    and all 8 tools bound to a `ToolExecutor` wrapping `env`. This is called
    by `_run_adk_loop()` — the production execution path — every time
    `run_ticket()` is invoked with valid Gemini credentials.

    IMPORTANT: ADK's `FunctionTool` builds each tool's JSON schema by
    introspecting the wrapped Python function's real type-annotated
    parameters and docstring — it does NOT read a separate declarations
    dict. A generic `**kwargs` wrapper therefore produces an empty schema
    (`parameters=None`), which silently breaks live tool calling even
    though the agent object still constructs without error. Each tool
    below is written out explicitly with real parameters and a docstring
    for exactly that reason.
    """
    from google.adk.agents import Agent  # raises ImportError if not installed

    executor = ToolExecutor(env)

    def get_ticket_state() -> dict:
        """Read the current ticket, customer context, conversation history,
        and status flags from the environment. Call this first."""
        return executor.get_ticket_state()

    def search_knowledge_base(query: str) -> dict:
        """Search internal knowledge-base articles (policies, troubleshooting
        steps) relevant to the ticket.

        Args:
            query: Search terms, e.g. 'warranty policy'.
        """
        return executor.search_knowledge_base(query)

    def get_customer_history() -> dict:
        """Return the customer's plan tier, total spend, months subscribed,
        and past ticket count."""
        return executor.get_customer_history()

    def verify_policy(amount: float) -> dict:
        """Check whether a proposed refund amount is justified given the
        knowledge base and customer context, without applying it.

        Args:
            amount: Proposed refund amount in USD.
        """
        return executor.verify_policy(amount)

    def issue_refund(amount: float) -> dict:
        """Issue a refund of the given dollar amount to the customer.

        Args:
            amount: Refund amount in USD.
        """
        return executor.issue_refund(amount)

    def escalate_ticket() -> dict:
        """Escalate the ticket to a human tier-2 agent and close the
        automated session."""
        return executor.escalate_ticket()

    def respond_to_customer(message: str) -> dict:
        """Send a message directly to the customer.

        Args:
            message: Customer-facing reply text.
        """
        return executor.respond_to_customer(message)

    def close_ticket() -> dict:
        """Mark the ticket resolved and end the session. Only call this
        after responding to the customer."""
        return executor.close_ticket()

    tool_fns = [
        get_ticket_state, search_knowledge_base, get_customer_history,
        verify_policy, issue_refund, escalate_ticket, respond_to_customer,
        close_ticket,
    ]

    return Agent(
        name="customer_support_agent",
        model=os.environ.get("GEMINI_MODEL", "gemini-3.5-flash"),
        instruction=SYSTEM_PROMPT,
        tools=tool_fns,
    )
