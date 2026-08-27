def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok"}


def test_reset_and_step_openenv_endpoints(client):
    res = client.post("/reset", json={"task_level": "easy"})
    assert res.status_code == 200
    session_id = res.json()["session_id"]

    res = client.post("/step", json={
        "session_id": session_id,
        "action": {"action_type": "SearchKB", "content": "password"},
    })
    assert res.status_code == 200
    assert "reward" in res.json()


def test_get_state_missing_session_returns_404(client):
    res = client.get("/state/does-not-exist")
    assert res.status_code == 404


def test_get_state_after_reset(client):
    res = client.post("/reset", json={"task_level": "medium"})
    session_id = res.json()["session_id"]
    res = client.get(f"/state/{session_id}")
    assert res.status_code == 200
    assert res.json()["state"]["task_id"] == "medium"


def test_agent_resolve_endpoint_returns_full_payload(client):
    res = client.post("/api/agent/resolve", json={"task_level": "hard"})
    assert res.status_code == 200
    body = res.json()
    for key in ["session_id", "status", "final_response", "reward", "metrics", "actions", "execution_path"]:
        assert key in body
    assert body["status"] in {"resolved", "escalated", "timed_out"}
    # No GOOGLE_API_KEY in the test environment (see conftest.py) -> the API
    # must transparently report it used the offline fallback, not silently
    # claim it used the production ADK path.
    assert body["execution_path"] == "deterministic_fallback"


def test_agent_session_and_trace_and_evaluation_endpoints(client):
    resolve_res = client.post("/api/agent/resolve", json={"task_level": "easy"})
    session_id = resolve_res.json()["session_id"]

    session_res = client.get(f"/api/agent/session/{session_id}")
    assert session_res.status_code == 200
    assert session_res.json()["ticket"]["id"] == "TKT-001"

    trace_res = client.get(f"/api/agent/trace/{session_id}")
    assert trace_res.status_code == 200
    assert len(trace_res.json()["actions"]) > 0

    eval_res = client.get(f"/api/agent/evaluation/{session_id}")
    assert eval_res.status_code == 200
    assert "resolution_success" in eval_res.json()["metrics"]


def test_agent_session_not_found_returns_404(client):
    res = client.get("/api/agent/session/nonexistent")
    assert res.status_code == 404
    res = client.get("/api/agent/evaluation/nonexistent")
    assert res.status_code == 404


def test_invalid_action_type_returns_422(client):
    client.post("/reset", json={"session_id": "s1", "task_level": "easy"})
    res = client.post("/step", json={
        "session_id": "s1",
        "action": {"action_type": "DeleteAccount"},
    })
    assert res.status_code == 422  # Pydantic Literal validation rejects unknown action types
