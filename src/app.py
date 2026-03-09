from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException

from src.api.schema import AskRequest, AskResponse, IngestRequest, IngestResponse
from src.services.rag_service import rag_service

logger = logging.getLogger("uvicorn")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up RAG API...")
    if rag_service.settings.auto_ingest_on_startup:
        try:
            result = rag_service.ingest(recreate=rag_service.settings.recreate_on_startup)
            logger.info("Startup ingest completed: %s", result)
        except Exception as exc:
            # Keep API alive even if startup ingest fails.
            logger.error("Startup ingest failed: %s", exc)
    yield
    logger.info("Shutting down RAG API...")


app = FastAPI(
    title="Phishing RAG API",
    description="RAG API to ingest phishing-related PDFs and answer questions with sources.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def read_root() -> dict[str, str]:
    return {"message": "Phishing RAG API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest) -> IngestResponse:
    try:
        result = rag_service.ingest(recreate=request.recreate)
        return IngestResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest) -> AskResponse:
    try:
        result = rag_service.ask(request.question)
        return AskResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc