import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Ensure tests never accidentally hit real Google Cloud services.
os.environ.setdefault("USE_FIRESTORE", "false")
os.environ.pop("GOOGLE_API_KEY", None)
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "false"

import pytest
from fastapi.testclient import TestClient

from server import app


@pytest.fixture()
def client():
    return TestClient(app)
