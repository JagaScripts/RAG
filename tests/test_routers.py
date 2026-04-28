from unittest.mock import AsyncMock
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)

def test_langchain_search_missing_query():
    # Debería dar 422 Unprocessable Entity si no enviamos la query
    response = client.post("/langchain/search")
    assert response.status_code == 422

def test_langchain_rag_missing_body():
    # Debería dar 422 Unprocessable Entity si el body está vacío
    response = client.post("/langchain/rag", json={})
    assert response.status_code == 422

def test_langchain_rag_mocked(mocker):
    # Mockear la cadena de Langchain para evitar consumir API real
    mock_result = {
        "question": "¿Qué es phishing?",
        "answer": "Es un ataque cibernético.",
        "source": AsyncMock(selection="dummy_source", reason="dummy_reason")
    }
    mock_result["source"].selection = "Documentación de seguridad"
    mock_result["source"].reason = "Coincidencia directa"
    
    # Mock ainvoke de rag_chain
    mocker.patch(
        "src.api.router_langchain.rag_chain.ainvoke",
        return_value=mock_result
    )

    response = client.post("/langchain/rag", json={"question": "¿Qué es phishing?"})
    
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "¿Qué es phishing?"
    assert data["answer"] == "Es un ataque cibernético."
    assert data["source"] == "Documentación de seguridad"
    assert data["source_reason"] == "Coincidencia directa"

def test_langchain_search_mocked(mocker):
    # Mockear Qdrant para devolver documentos dummy
    dummy_doc = type("Document", (), {"page_content": "Phishing text", "metadata": {}})()
    
    mocker.patch(
        "src.api.router_langchain.qdrant_langchain.asimilarity_search",
        return_value=[dummy_doc]
    )

    response = client.post("/langchain/search?query=phishing")
    
    assert response.status_code == 200
    assert len(response.json()) > 0
