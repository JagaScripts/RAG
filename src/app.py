"""FastAPI application for the Phishing RAG system.

Defines endpoints for:
- Health checks
- Document ingestion into Qdrant vector store
- Question answering with source attribution
"""

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.services.rag_service import rag_service


logger = logging.getLogger("uvicorn")


class IngestRequest(BaseModel):
    """Request model for document ingestion.
    
    Attributes:
        recreate: If True, recreates the Qdrant collection; if False, appends to existing.
    """
    recreate: bool = Field(default=True, description="If true, recreate the collection before indexing")


class AskRequest(BaseModel):
    """Request model for phishing knowledge queries.
    
    Attributes:
        question: The question to ask the RAG system about phishing.
    """
    question: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage API startup and shutdown events.
    
    On startup: Optionally ingests documents into Qdrant (if AUTO_INGEST_ON_STARTUP=true).
    On shutdown: Logs shutdown event.
    """
    logger.info("Starting Phishing RAG API")
    if rag_service.settings.auto_ingest_on_startup:
        try:
            rag_service.ingest(recreate=rag_service.settings.recreate_on_startup)
            logger.info("Auto ingest completed")
        except Exception as exc:
            logger.warning("Auto ingest skipped: %s", exc)
    yield
    logger.info("Shutting down Phishing RAG API")


app = FastAPI(
    title="Phishing Knowledge RAG API",
    description="Single RAG API with LangChain + Qdrant over phishing PDFs",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def read_root() -> dict[str, str]:
    """Root endpoint returning API status."""
    return {"message": "Phishing RAG API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict[str, str | int | list[str]]:
    """Ingest PDF documents into the Qdrant vector store.
    
    Processes PDFs or pre-chunked documents, generates embeddings using Google Gemini,
    and stores them in Qdrant for similarity search.
    
    Args:
        request: IngestRequest with recreate flag.
        
    Returns:
        Dictionary with collection info, chunk count, file names, and source URLs.
    """
    try:
        return rag_service.ingest(recreate=request.recreate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, str | list[str]]:
    """Answer a question using the RAG system.
    
    Retrieves relevant document chunks from Qdrant, generates a response using
    Google Gemini, and includes source URLs for attribution.
    
    Args:
        request: AskRequest with the phishing question.
        
    Returns:
        Dictionary with generated answer and list of source URLs.
    """
    try:
        return rag_service.ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ask failed: {exc}") from exc