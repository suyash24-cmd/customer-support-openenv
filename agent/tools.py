"""
tools.py — Tool layer between the Gemini/ADK agent and the OpenEnv
CustomerSupportEnv.

Every tool call is translated into exactly one `env.step(Action(...))` call
(or a read of `env.get_state()`), so the agent can never bypass the
environment's reward/grading logic. This module also defines the Gemini
function-calling declarations used by agent.py.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from env.environment import CustomerSupportEnv
from env.models import Action


# ---------------------------------------------------------------------------
# Reference tool contract (plain dicts). The production ADK agent
# (agent/agent.py::build_adk_agent) does NOT read this — ADK generates each
# tool's real JSON schema by introspecting the typed Python functions defined
# there directly. This dict is kept as a single human-readable reference of
# the 8-tool contract for documentation/tests, not as an executable schema.
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS: list[Dict[str, Any]] = [
    {
        "name": "get_ticket_state",
        "description": "Read the current ticket, customer context, conversation "
        "history, and status flags from the environment. Call this first.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "search_knowledge_base",
        "description": "Search internal knowledge-base articles (policies, "
        "troubleshooting steps) relevant to the ticket.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search terms, e.g. 'warranty policy'."}
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_customer_history",
        "description": "Return the customer's plan tier, total spend, months "
        "subscribed, and past ticket count from the current observation.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "verify_policy",
        "description": "Check whether a proposed refund amount is justified "
        "given the knowledge base and customer context, without applying it.",
        "parameters": {
            "type": "object",
            "properties": {
                "amount": {"type": "number", "description": "Proposed refund amount in USD."}
            },
            "required": ["amount"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund of the given dollar amount to the customer.",
        "parameters": {
            "type": "object",
            "properties": {"amount": {"type": "number", "description": "Refund amount in USD."}},
            "required": ["amount"],
        },
    },
    {
        "name": "escalate_ticket",
        "description": "Escalate the ticket to a human tier-2 agent and close "
        "the automated session.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "respond_to_customer",
        "description": "Send a message directly to the customer.",
        "parameters": {
            "type": "object",
            "properties": {"message": {"type": "string", "description": "Customer-facing reply text."}},
            "required": ["message"],
        },
    },
    {
        "name": "close_ticket",
        "description": "Mark the ticket resolved and end the session. Only "
        "call this after responding to the customer.",
        "parameters": {"type": "object", "properties": {}},
    },
]


class ToolExecutor:
    """Executes tool calls against a live CustomerSupportEnv instance."""

    def __init__(self, env: CustomerSupportEnv):
        self.env = env

    # -- read-only tools ---------------------------------------------------
    def get_ticket_state(self) -> Dict[str, Any]:
        state = self.env.get_state()
        return {
            "ticket": state.ticket.model_dump(),
            "is_escalated": state.is_escalated,
            "is_closed": state.is_closed,
            "conversation_history": state.conversation_history,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
        }

    def get_customer_history(self) -> Dict[str, Any]:
        state = self.env.get_state()
        return state.customer_context.model_dump()

    def verify_policy(self, amount: float) -> Dict[str, Any]:
        """Non-mutating policy check — reasons over the KB without stepping the env."""
        state = self.env.get_state()
        kb = state.kb_articles
        justified = False
        reason = "No matching policy found; refund not automatically justified."
        if state.task_id == "medium" and abs(amount - 150.0) < 0.01:
            justified = True
            reason = kb.get("warranty_policy", "Warranty policy covers this refund.")
        elif state.task_id == "hard" and amount > 0:
            justified = True
            reason = kb.get("billing_issues", "Billing policy covers duplicate-charge refunds.")
        elif state.task_id == "easy":
            justified = False
            reason = kb.get("password_reset", "Password issues are not refund-eligible.")
        return {"amount": amount, "justified": justified, "reason": reason}

    # -- mutating tools (each maps 1:1 onto an env.step Action) -------------
    def search_knowledge_base(self, query: str) -> Dict[str, Any]:
        obs, reward, done, info = self.env.step(Action(action_type="SearchKB", content=query))
        return self._package(obs, reward, done, info)

    def issue_refund(self, amount: float) -> Dict[str, Any]:
        obs, reward, done, info = self.env.step(Action(action_type="IssueRefund", amount=amount))
        return self._package(obs, reward, done, info)

    def escalate_ticket(self) -> Dict[str, Any]:
        obs, reward, done, info = self.env.step(Action(action_type="Escalate"))
        return self._package(obs, reward, done, info)

    def respond_to_customer(self, message: str) -> Dict[str, Any]:
        obs, reward, done, info = self.env.step(Action(action_type="Reply", content=message))
        return self._package(obs, reward, done, info)

    def close_ticket(self) -> Dict[str, Any]:
        obs, reward, done, info = self.env.step(Action(action_type="CloseTicket"))
        return self._package(obs, reward, done, info)

    @staticmethod
    def _package(obs, reward: float, done: bool, info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "feedback": obs.last_action_feedback,
            "reward": reward,
            "done": done,
            "is_escalated": obs.is_escalated,
            "is_closed": obs.is_closed,
            "info": info,
        }

    def dispatch(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        handlers: Dict[str, Callable[..., Dict[str, Any]]] = {
            "get_ticket_state": self.get_ticket_state,
            "get_customer_history": self.get_customer_history,
            "verify_policy": self.verify_policy,
            "search_knowledge_base": self.search_knowledge_base,
            "issue_refund": self.issue_refund,
            "escalate_ticket": self.escalate_ticket,
            "respond_to_customer": self.respond_to_customer,
            "close_ticket": self.close_ticket,
        }
        if name not in handlers:
            raise ValueError(f"Unknown tool: {name}")
        return handlers[name](**args)


# Human-readable, safe trace lines shown in the UI/API — never chain-of-thought.
TRACE_LABELS: Dict[str, str] = {
    "get_ticket_state": "Analyzed ticket",
    "get_customer_history": "Checked customer history",
    "verify_policy": "Verified policy",
    "search_knowledge_base": "Searched knowledge base",
    "issue_refund": "Issued refund",
    "escalate_ticket": "Escalated to human support",
    "respond_to_customer": "Generated customer response",
    "close_ticket": "Closed ticket",
}
