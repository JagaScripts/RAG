"""Tests de integración y unitarios para el RAG de phishing.

Incluye:
- Tests de endpoints FastAPI
- Tests de validación de modelos
- Tests de funciones helper de rag_service
- Tests mock para RAGService
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
import os

from src.app import app, IngestRequest, AskRequest
from src.services.rag_service import _normalize, _as_bool, RAGService


# ============================================================================
# CLIENT DE PRUEBA
# ============================================================================

client = TestClient(app)


# ============================================================================
# TESTS DE ENDPOINTS
# ============================================================================

class TestHealthEndpoint:
    """Tests para el endpoint /health"""

    def test_health_returns_ok(self):
        """Verifica que /health retorna status ok"""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestRootEndpoint:
    """Tests para el endpoint raíz /"""

    def test_root_returns_message(self):
        """Verifica que / retorna mensaje de API en ejecución"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "running" in data["message"].lower()


class TestIngestEndpoint:
    """Tests para el endpoint /ingest"""

    @patch("src.services.rag_service.rag_service.ingest")
    def test_ingest_with_recreate_true(self, mock_ingest):
        """Verifica que /ingest funciona con recreate=True"""
        mock_ingest.return_value = {
            "collection_name": "phishing_knowledge",
            "indexed_chunks": 44,
            "indexed_files": ["doc1.pdf"],
            "indexed_urls": ["https://example.com"],
        }

        response = client.post("/ingest", json={"recreate": True})
        assert response.status_code == 200
        data = response.json()
        assert data["collection_name"] == "phishing_knowledge"
        assert data["indexed_chunks"] == 44
        mock_ingest.assert_called_once_with(recreate=True)

    @patch("src.services.rag_service.rag_service.ingest")
    def test_ingest_with_recreate_false(self, mock_ingest):
        """Verifica que /ingest funciona con recreate=False"""
        mock_ingest.return_value = {
            "collection_name": "phishing_knowledge",
            "indexed_chunks": 10,
            "indexed_files": ["new_doc.pdf"],
            "indexed_urls": [],
        }

        response = client.post("/ingest", json={"recreate": False})
        assert response.status_code == 200
        mock_ingest.assert_called_once_with(recreate=False)

    @patch("src.services.rag_service.rag_service.ingest")
    def test_ingest_handles_error(self, mock_ingest):
        """Verifica que /ingest maneja errores correctamente"""
        mock_ingest.side_effect = ValueError("Data directory does not exist")

        response = client.post("/ingest", json={"recreate": True})
        assert response.status_code == 400
        assert "Data directory" in response.json()["detail"]


class TestAskEndpoint:
    """Tests para el endpoint /ask"""

    @patch("src.services.rag_service.rag_service.ask")
    def test_ask_with_valid_question(self, mock_ask):
        """Verifica que /ask responde preguntas válidas"""
        mock_ask.return_value = {
            "answer": "El phishing es un ataque cibernético...\n\nFuentes:\n- https://example.com",
            "sources": ["https://example.com"],
        }

        response = client.post("/ask", json={"question": "¿Qué es phishing?"})
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) == 1
        mock_ask.assert_called_once_with("¿Qué es phishing?")

    def test_ask_with_empty_question(self):
        """Verifica que /ask rechaza preguntas vacías"""
        response = client.post("/ask", json={"question": ""})
        # FastAPI retorna 422 cuando Pydantic rechaza los datos
        assert response.status_code == 422

    @patch("src.services.rag_service.rag_service.ask")
    def test_ask_handles_error(self, mock_ask):
        """Verifica que /ask maneja errores correctamente"""
        mock_ask.side_effect = Exception("Qdrant connection failed")

        response = client.post("/ask", json={"question": "test"})
        assert response.status_code == 500


# ============================================================================
# TESTS DE MODELOS PYDANTIC
# ============================================================================

class TestIngestRequestModel:
    """Tests para el modelo IngestRequest"""

    def test_ingest_request_with_recreate_true(self):
        """Verifica que IngestRequest acepta recreate=True"""
        req = IngestRequest(recreate=True)
        assert req.recreate is True

    def test_ingest_request_with_recreate_false(self):
        """Verifica que IngestRequest acepta recreate=False"""
        req = IngestRequest(recreate=False)
        assert req.recreate is False

    def test_ingest_request_default_recreate(self):
        """Verifica que recreate por defecto es True"""
        req = IngestRequest()
        assert req.recreate is True


class TestAskRequestModel:
    """Tests para el modelo AskRequest"""

    def test_ask_request_with_valid_question(self):
        """Verifica que AskRequest acepta preguntas válidas"""
        req = AskRequest(question="¿Qué es phishing?")
        assert req.question == "¿Qué es phishing?"

    def test_ask_request_rejects_empty_question(self):
        """Verifica que AskRequest rechaza preguntas vacías"""
        with pytest.raises(ValueError):
            AskRequest(question="")


# ============================================================================
# TESTS DE FUNCIONES HELPER
# ============================================================================

class TestNormalizeFunction:
    """Tests para la función _normalize"""

    def test_normalize_lowercase(self):
        """Verifica que _normalize convierte a minúsculas"""
        assert _normalize("PHISHING") == "phishing"

    def test_normalize_phising_typo(self):
        """Verifica que _normalize corrige 'phising' a 'phishing'"""
        assert "phishing" in _normalize("phising")

    def test_normalize_removes_diacritics(self):
        """Verifica que _normalize elimina diacríticos"""
        result = _normalize("Información de Ciberseguridad")
        assert "a" in result
        assert "ó" not in result

    def test_normalize_removes_special_chars(self):
        """Verifica que _normalize elimina caracteres especiales"""
        result = _normalize("PDF-Guía_Phishing@2024")
        assert "@" not in result
        assert "-" not in result

    def test_normalize_spaces_consistent(self):
        """Verifica que _normalize normaliza espacios"""
        result = _normalize("Phishing   Attack    Guide")
        assert "  " not in result


class TestAsBoolFunction:
    """Tests para la función _as_bool"""

    def test_as_bool_with_true_values(self):
        """Verifica que _as_bool reconoce valores verdaderos"""
        assert _as_bool("1", False) is True
        assert _as_bool("true", False) is True
        assert _as_bool("True", False) is True
        assert _as_bool("yes", False) is True
        assert _as_bool("YES", False) is True
        assert _as_bool("y", False) is True
        assert _as_bool("on", False) is True

    def test_as_bool_with_false_values(self):
        """Verifica que _as_bool reconoce valores falsos"""
        assert _as_bool("0", True) is False
        assert _as_bool("false", True) is False
        assert _as_bool("no", True) is False
        assert _as_bool("anything_else", True) is False

    def test_as_bool_with_none_returns_default(self):
        """Verifica que _as_bool retorna default si value es None"""
        assert _as_bool(None, True) is True
        assert _as_bool(None, False) is False


# ============================================================================
# TESTS DE RAGSERVICE
# ============================================================================

class TestRAGServiceSettings:
    """Tests para la configuración de RAGService"""

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"})
    def test_ragservice_loads_settings(self):
        """Verifica que RAGService carga configuración desde env"""
        with patch.object(RAGService, "_ensure_qdrant_connection"):
            service = RAGService()
            assert service.settings.google_api_key == "test-key"

    @patch.dict(os.environ, {}, clear=True)
    def test_ragservice_uses_defaults(self):
        """Verifica que RAGService usa valores por defecto"""
        with patch.object(RAGService, "_ensure_qdrant_connection"):
            service = RAGService()
            assert service.settings.qdrant_url == "http://localhost:6333"
            assert service.settings.collection_name == "phishing_knowledge"


class TestRAGServiceHelpers:
    """Tests para métodos helper de RAGService"""

    @patch.object(RAGService, "_ensure_qdrant_connection")
    def test_ensure_models_raises_without_api_key(self, mock_conn):
        """Verifica que _ensure_models falla sin API key"""
        service = RAGService()
        service.settings.google_api_key = ""

        with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
            service._ensure_models()

    @patch.object(RAGService, "_ensure_qdrant_connection")
    @patch("src.services.rag_service.GoogleGenerativeAIEmbeddings")
    def test_ensure_models_lazy_loads(self, mock_embeddings, mock_conn):
        """Verifica que _ensure_models hace lazy-loading"""
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
            service = RAGService()
            assert service.embeddings is None

            service._ensure_models()
            # Verificar que se intentó crear los modelos
            assert service.embeddings is not None or mock_embeddings.called


# ============================================================================
# TESTS DE INTEGRACIÓN (MOCK)
# ============================================================================

class TestRAGIntegration:
    """Tests de integración del flujo completo RAG (con mocks)"""

    @patch("src.services.rag_service.rag_service.ask")
    def test_full_ask_workflow(self, mock_ask):
        """Verifica el flujo completo de pregunta"""
        expected_response = {
            "answer": "El phishing es un ataque que suplanta identidades...\n\nFuentes:\n- https://source1.pdf\n- https://source2.pdf",
            "sources": ["https://source1.pdf", "https://source2.pdf"],
        }
        mock_ask.return_value = expected_response

        response = client.post("/ask", json={"question": "¿Cómo evitar phishing?"})

        assert response.status_code == 200
        data = response.json()
        assert len(data["sources"]) == 2
        assert "Fuentes:" in data["answer"]

    @patch("src.services.rag_service.rag_service.ingest")
    def test_full_ingest_workflow(self, mock_ingest):
        """Verifica el flujo completo de ingesta"""
        expected_response = {
            "collection_name": "phishing_knowledge",
            "indexed_pages": 100,
            "indexed_chunks": 50,
            "indexed_files": ["guia1.pdf", "guia2.pdf", "guia3.pdf"],
            "indexed_urls": [
                "https://source1.pdf",
                "https://source2.pdf",
                "https://source3.pdf",
            ],
        }
        mock_ingest.return_value = expected_response

        response = client.post("/ingest", json={"recreate": True})

        assert response.status_code == 200
        data = response.json()
        assert data["indexed_chunks"] == 50
        assert len(data["indexed_files"]) == 3
