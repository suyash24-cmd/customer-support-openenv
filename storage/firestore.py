"""
storage/firestore.py — Persistence layer for sessions, traces, and
evaluation results.

Uses Google Cloud Firestore when USE_FIRESTORE=true (and credentials /
GOOGLE_CLOUD_PROJECT are available). Otherwise falls back to a process-local
in-memory store so the app runs with zero Google Cloud setup — required by
CRITICAL_IMPLEMENTATION_RULES #10 ("keep the project runnable locally").

The public API (`SessionStore`) is identical in both modes, so callers
(server.py) never need to know which backend is active.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional


class _InMemoryBackend:
    """Local fallback store. Data is lost on process restart — fine for
    local dev and for the pytest suite, which never needs persistence."""

    def __init__(self) -> None:
        self._data: Dict[str, Dict[str, Any]] = {}

    def set(self, collection: str, doc_id: str, value: Dict[str, Any]) -> None:
        self._data.setdefault(collection, {})[doc_id] = value

    def get(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        return self._data.get(collection, {}).get(doc_id)

    def list_ids(self, collection: str) -> list[str]:
        return list(self._data.get(collection, {}).keys())


class _FirestoreBackend:
    """Real Firestore-backed store. Only imported/instantiated when
    USE_FIRESTORE=true, so `google-cloud-firestore` credentials are never
    required for local development."""

    def __init__(self, prefix: str) -> None:
        from google.cloud import firestore  # deferred import

        self._client = firestore.Client()
        self._prefix = prefix

    def _collection(self, name: str):
        return self._client.collection(f"{self._prefix}_{name}")

    def set(self, collection: str, doc_id: str, value: Dict[str, Any]) -> None:
        self._collection(collection).document(doc_id).set(value)

    def get(self, collection: str, doc_id: str) -> Optional[Dict[str, Any]]:
        snap = self._collection(collection).document(doc_id).get()
        return snap.to_dict() if snap.exists else None

    def list_ids(self, collection: str) -> list[str]:
        return [doc.id for doc in self._collection(collection).stream()]


class SessionStore:
    """Stores tickets, conversation history, agent action traces, refunds,
    and evaluation results, keyed by session_id.

    Collections used: tickets, conversation_history, agent_actions,
    refunds, evaluation_results, customer_context.
    """

    def __init__(self) -> None:
        use_firestore = os.environ.get("USE_FIRESTORE", "false").lower() == "true"
        prefix = os.environ.get("FIRESTORE_COLLECTION_PREFIX", "customer_support")
        if use_firestore:
            try:
                self._backend = _FirestoreBackend(prefix)
                self.backend_name = "firestore"
                return
            except Exception:
                # Firestore unreachable/misconfigured — degrade gracefully
                # rather than crashing the whole service.
                pass
        self._backend = _InMemoryBackend()
        self.backend_name = "memory"

    def save_session(self, session_id: str, run_result_dict: Dict[str, Any]) -> None:
        self._backend.set("tickets", session_id, run_result_dict.get("ticket", {}))
        self._backend.set("conversation_history", session_id, {
            "history": run_result_dict.get("conversation_history", [])
        })
        self._backend.set("agent_actions", session_id, {"trace": run_result_dict.get("actions", [])})
        self._backend.set("refunds", session_id, {"amount": run_result_dict.get("refund_issued", 0.0)})
        self._backend.set("evaluation_results", session_id, run_result_dict.get("metrics", {}))
        self._backend.set("customer_context", session_id, run_result_dict.get("customer_context", {}))

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        ticket = self._backend.get("tickets", session_id)
        if ticket is None:
            return None
        return {
            "ticket": ticket,
            "conversation_history": (self._backend.get("conversation_history", session_id) or {}).get("history", []),
            "actions": (self._backend.get("agent_actions", session_id) or {}).get("trace", []),
            "refund_issued": (self._backend.get("refunds", session_id) or {}).get("amount", 0.0),
            "metrics": self._backend.get("evaluation_results", session_id) or {},
            "customer_context": self._backend.get("customer_context", session_id) or {},
        }

    def get_evaluation(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._backend.get("evaluation_results", session_id)

    def list_sessions(self) -> list[str]:
        return self._backend.list_ids("tickets")
