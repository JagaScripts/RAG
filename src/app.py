from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.services.rag_service import rag_service


logger = logging.getLogger("uvicorn")


class IngestRequest(BaseModel):
    recreate: bool = Field(default=True, description="If true, recreate the collection before indexing")


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    return {"message": "Phishing RAG API is running"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict[str, str | int | list[str]]:
    try:
        return rag_service.ingest(recreate=request.recreate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, str | list[str]]:
    try:
        return rag_service.ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ask failed: {exc}") from exc