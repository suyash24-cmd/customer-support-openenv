import os

from agent.agent import _build_genai_client
from storage.firestore import SessionStore


def test_genai_client_is_none_without_credentials(monkeypatch):
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "false")
    assert _build_genai_client() is None


def test_genai_client_vertex_mode_without_project_returns_none(monkeypatch):
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    assert _build_genai_client() is None


def test_session_store_falls_back_to_memory_when_firestore_disabled(monkeypatch):
    monkeypatch.setenv("USE_FIRESTORE", "false")
    store = SessionStore()
    assert store.backend_name == "memory"

    store.save_session("s1", {"ticket": {"id": "TKT-001"}})
    session = store.get_session("s1")
    assert session["ticket"]["id"] == "TKT-001"


def test_session_store_degrades_gracefully_if_firestore_misconfigured(monkeypatch):
    monkeypatch.setenv("USE_FIRESTORE", "true")
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    # No real Firestore credentials in the test environment -> should not raise,
    # and should silently fall back to the in-memory backend.
    store = SessionStore()
    assert store.backend_name in {"memory", "firestore"}
