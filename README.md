# Phishing RAG API

Repositorio minimo para ejecutar un RAG sobre phishing con:

- FastAPI
- LangChain
- Qdrant
- Google Gemini

El endpoint `POST /ask` devuelve siempre respuesta + bloque final `Fuentes:` con URLs.

## Estructura

- `src/app.py`: API y endpoints
- `src/main.py`: arranque local
- `src/services/rag_service.py`: ingesta, retrieval y respuesta
- `docker_compose.yml`: Qdrant + API
- `qdrant_config/config.yml`: config de Qdrant
- `.env.example`: variables de entorno
- `URLs base conocimiento.txt`: mapeo documento -> URL fuente

## Requisitos

- Python 3.11 o 3.12
- Docker Desktop en ejecución
- API key de Google (`GOOGLE_API_KEY`)

## Configuracion

1. Crear `.env`:

```powershell
Copy-Item .env.example .env
```

1. Editar `.env` y completar al menos:

```env
GOOGLE_API_KEY="tu_api_key"
QDRANT_URL="http://localhost:6333"
QDRANT_COLLECTION="phishing_knowledge"
DATA_DIR="data"
CHUNKS_DIR="data/optimized_chunks"
KNOWLEDGE_URLS_FILE="URLs base conocimiento.txt"
SIMILARITY_TOP_K="5"
CHUNK_SIZE="1200"
CHUNK_OVERLAP="200"
AUTO_INGEST_ON_STARTUP="true"
RECREATE_ON_STARTUP="true"
```

## Ejecucion local

1. Levantar Qdrant:

```bash
docker compose -f docker_compose.yml up -d qdrant
```

1. Instalar dependencias:

```bash
pip install .
```

1. Arrancar API:

```bash
python -m src.main
```

1. Probar estado:

```bash
curl http://localhost:8000/health
```

1. Ingestar (si procede):

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"recreate": true}'
```

1. Preguntar al RAG:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Que es el phishing y como protegerse?"}'
```

## Endpoints


## Testing

Suite de tests con 27 tests unitarios e integración (pytest).

### Ejecutar todos los tests:

```bash
uv run pytest tests/ -v
```

### Ejecutar tests de una clase específica:

```bash
uv run pytest tests/test_rag.py::TestHealthEndpoint -v
```

### Tests incluidos:

- **Endpoints**: Validación de `/health`, `/ingest`, `/ask`
- **Modelos**: Validación de `IngestRequest` y `AskRequest`
- **Funciones Helper**: `_normalize`, `_as_bool`
- **RAGService**: Configuración, modelos lazy-loading
- **Integración**: Flujos completos con mocks

Todos los tests usan mocks para no requerir Qdrant ni API key en ejecución.
