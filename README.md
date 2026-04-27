# Phishing Knowledge RAG API

Proyecto RAG centrado en phishing usando:

- LangChain para retrieval + generacion
- Qdrant como base de datos vectorial
- Google Gemini (API KEY de Google) para embeddings y chat
- 3 PDFs en `data/` como base de conocimiento
- `URLs base conocimiento.txt` para mostrar la fuente URL en cada respuesta

La API devuelve una unica respuesta textual y, al final, el bloque `Fuentes:` con las URLs de donde se obtuvo la informacion.

## Estructura funcional

- `src/services/rag_service.py`: servicio principal (ingesta, recuperacion, respuesta y fuentes)
- `src/app.py`: API FastAPI con `/ingest` y `/ask`
- `scripts/create_langchain_index.py`: indexado manual
- `scripts/routing_generation.py`: consulta por CLI

## Requisitos

- Python 3.11+
- Docker Desktop (para Qdrant)
- `GOOGLE_API_KEY`

## Configuracion

1. Crea `.env` desde plantilla:

```powershell
Copy-Item .env.example .env
```

1. Edita `.env` y define al menos:

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

1. Verifica que existan los 3 PDFs en `data/` y que `URLs base conocimiento.txt` tenga una URL por documento.

## Arranque rapido (local)

1. Levanta Qdrant:

```bash
docker compose -f docker_compose.yml up -d qdrant
```

1. Instala dependencias:

```bash
pip install .
```

1. Arranca la API:

```bash
python -m src.main
```

1. Indexa conocimiento (si no usas auto-ingest):

```bash
curl -X POST http://localhost:8000/ingest -H "Content-Type: application/json" -d '{"recreate": true}'
```

1. Pregunta sobre phishing:

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question": "Que es el phishing y como protegerse?"}'
```

## Endpoints

- `GET /`: estado API
- `GET /health`: healthcheck
- `POST /ingest`: indexa PDFs (`{"recreate": true|false}`)
- `POST /ask`: responde pregunta (`{"question": "..."}`)

## Scripts utiles

- `python -m scripts.create_langchain_index`: indexar desde terminal
- `python -m scripts.routing_generation "tu pregunta"`: consultar por CLI
- `powershell -ExecutionPolicy Bypass -File .\scripts\run_full_pipeline.ps1`: pipeline en Windows

## Nota Windows + Docker

Si aparece `WinError 10061`, Docker Desktop no esta levantado. Arrancalo y repite:

```powershell
docker compose -f docker_compose.yml up -d qdrant
```
