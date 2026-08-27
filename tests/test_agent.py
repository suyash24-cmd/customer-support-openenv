import pytest

from agent.agent import run_ticket
from agent.evaluator import evaluate
from agent.tools import ToolExecutor
from env.environment import CustomerSupportEnv


def test_tool_executor_dispatches_known_tools():
    env = CustomerSupportEnv(task_level="medium")
    env.reset()
    executor = ToolExecutor(env)

    state = executor.get_ticket_state()
    assert state["ticket"]["id"] == "TKT-002"

    kb_result = executor.dispatch("search_knowledge_base", {"query": "warranty"})
    assert "feedback" in kb_result

    policy = executor.verify_policy(150.0)
    assert policy["justified"] is True


def test_tool_executor_rejects_unknown_tool():
    env = CustomerSupportEnv(task_level="easy")
    env.reset()
    executor = ToolExecutor(env)
    with pytest.raises(ValueError):
        executor.dispatch("delete_customer_account", {})


@pytest.mark.parametrize("task_level,expected_status", [
    ("easy", "resolved"),
    ("medium", "resolved"),
    ("hard", "resolved"),
])
def test_deterministic_agent_loop_resolves_without_gemini_credentials(task_level, expected_status):
    """No GOOGLE_API_KEY is set (see conftest.py), so this exercises the
    oracle fallback path end-to-end against the real environment."""
    result = run_ticket(session_id="test-session", task_level=task_level)
    assert result.status == expected_status
    assert len(result.trace) > 0
    assert result.final_response  # every scenario should reply to the customer


def test_evaluator_produces_expected_metric_keys():
    result = run_ticket(session_id="test-eval", task_level="medium")
    metrics = evaluate(result)
    for key in [
        "resolution_success", "policy_compliance", "tool_efficiency",
        "reward", "num_actions", "escalation_rate", "refund_accuracy", "status",
    ]:
        assert key in metrics


def test_evaluator_refund_accuracy_matches_correct_medium_refund():
    result = run_ticket(session_id="test-refund", task_level="medium")
    metrics = evaluate(result)
    assert metrics["refund_accuracy"] == 1.0


REQUIRED_TOOL_NAMES = {
    "get_ticket_state", "get_customer_history", "verify_policy",
    "search_knowledge_base", "issue_refund", "escalate_ticket",
    "respond_to_customer", "close_ticket",
}


def test_adk_agent_constructs_with_gemini_model_and_all_tools_registered():
    """Verifies the ADK agent can be constructed and that all 8 required
    tools are registered by name — no Gemini/network call involved."""
    from agent.agent import build_adk_agent

    env = CustomerSupportEnv(task_level="easy")
    env.reset()
    adk_agent = build_adk_agent(env)

    assert adk_agent.name == "customer_support_agent"
    assert adk_agent.model == "gemini-3.5-flash"
    registered_names = {t.__name__ for t in adk_agent.tools}
    assert registered_names == REQUIRED_TOOL_NAMES


def test_run_ticket_uses_real_adk_runner_as_production_path(monkeypatch):
    """Mocks Runner.run (no network/credentials) to verify run_ticket()
    actually routes through the Google ADK Runner — not the deterministic
    fallback — whenever Gemini credentials are configured."""
    from google.adk.runners import Runner
    from google.genai import types

    from agent.agent import run_ticket

    def fake_run(self, *, user_id, session_id, new_message, **kwargs):
        # Simulate one ADK turn: the model calls search_knowledge_base,
        # then issue_refund, then respond_to_customer, then close_ticket.
        agent_tools = {t.__name__: t for t in self.agent.tools}

        def make_event(name, args):
            call = types.FunctionCall(id=f"call-{name}", name=name, args=args)
            call_event = type("FakeEvent", (), {
                "content": types.Content(role="model", parts=[types.Part(function_call=call)]),
                "author": "model",
            })()
            result = agent_tools[name](**args)
            response = types.FunctionResponse(id=f"call-{name}", name=name, response=result)
            response_event = type("FakeEvent", (), {
                "content": types.Content(role="tool", parts=[types.Part(function_response=response)]),
                "author": name,
            })()
            return call_event, response_event

        for call_evt, resp_evt in [
            make_event("search_knowledge_base", {"query": "warranty policy"}),
            make_event("issue_refund", {"amount": 150.0}),
            make_event("respond_to_customer", {"message": "Refund issued for your defective router."}),
            make_event("close_ticket", {}),
        ]:
            yield call_evt
            yield resp_evt

    monkeypatch.setattr(Runner, "run", fake_run)
    # Fake credentials so run_ticket() takes the ADK path, without ever
    # calling out to the real Gemini API.
    fake_client = object()

    result = run_ticket(session_id="mocked-adk-session", task_level="medium", client=fake_client)

    assert result.execution_path == "adk"
    assert result.status == "resolved"
    assert result.final_response == "Refund issued for your defective router."
    tool_names_called = [t.tool for t in result.trace]
    assert tool_names_called == [
        "search_knowledge_base", "issue_refund", "respond_to_customer", "close_ticket",
    ]
    # Reward must come from the REAL environment (via the real tool
    # functions executed inside fake_run), not from a hardcoded number.
    assert result.total_reward > 0
    assert result.env.get_state().refund_issued == 150.0


def test_build_adk_agent_produces_real_callable_tools_with_schemas():
    """Guards against a regression to `**kwargs`-only tool wrappers, which
    build successfully but silently produce empty ADK tool schemas
    (no parameters, no description) and therefore never work in a live
    agent run. See the docstring on build_adk_agent for the full story."""
    from google.adk.tools import FunctionTool

    from agent.agent import build_adk_agent

    env = CustomerSupportEnv(task_level="medium")
    env.reset()
    adk_agent = build_adk_agent(env)

    assert adk_agent.name == "customer_support_agent"
    assert len(adk_agent.tools) == 8

    search_fn = next(t for t in adk_agent.tools if t.__name__ == "search_knowledge_base")
    declaration = FunctionTool(search_fn)._get_declaration()
    assert declaration.description  # must not be None/empty
    assert declaration.parameters_json_schema is not None
    assert "query" in declaration.parameters_json_schema["properties"]

    # Calling the tool the way ADK's Runner would must mutate the *real*
    # OpenEnv environment, not a mock.
    result = search_fn(query="warranty policy")
    assert "feedback" in result
    assert env.get_state().kb_queries_made == 1

    refund_fn = next(t for t in adk_agent.tools if t.__name__ == "issue_refund")
    refund_fn(amount=150.0)
    assert env.get_state().refund_issued == 150.0
