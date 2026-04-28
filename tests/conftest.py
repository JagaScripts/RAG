"""Configuración compartida para tests de pytest.

Define fixtures y configuraciones globales para todos los tests.
"""

import pytest
from unittest.mock import patch, MagicMock
import os


@pytest.fixture(scope="session", autouse=True)
def setup_test_env():
    """Configura variables de entorno para tests."""
    os.environ["GOOGLE_API_KEY"] = "test-api-key-for-testing"
    os.environ["QDRANT_URL"] = "http://localhost:6333"
    os.environ["QDRANT_COLLECTION"] = "test_phishing_knowledge"
    os.environ["DATA_DIR"] = "data"
    os.environ["CHUNKS_DIR"] = "data/optimized_chunks"
    os.environ["AUTO_INGEST_ON_STARTUP"] = "false"
    os.environ["RECREATE_ON_STARTUP"] = "false"
    yield
    # Cleanup si es necesario


@pytest.fixture
def mock_qdrant_client():
    """Mock para QdrantClient."""
    with patch("src.services.rag_service.QdrantClient") as mock:
        client = MagicMock()
        client.collection_exists.return_value = True
        mock.return_value = client
        yield mock


@pytest.fixture
def mock_embeddings():
    """Mock para GoogleGenerativeAIEmbeddings."""
    with patch("src.services.rag_service.GoogleGenerativeAIEmbeddings") as mock:
        yield mock


@pytest.fixture
def mock_llm():
    """Mock para ChatGoogleGenerativeAI."""
    with patch("src.services.rag_service.ChatGoogleGenerativeAI") as mock:
        llm = MagicMock()
        llm.invoke.return_value.content = "Respuesta de prueba"
        mock.return_value = llm
        yield mock
