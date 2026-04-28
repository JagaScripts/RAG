"""Aplicación FastAPI para el sistema RAG de phishing.

Define:
- Modelos de solicitud (IngestRequest, AskRequest)
- Contexto de ciclo de vida (lifespan) para auto-ingesta de documentos
- Endpoints HTTP (/health, /ingest, /ask)
"""
from contextlib import asynccontextmanager
import logging
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.services.rag_service import rag_service


logger = logging.getLogger("uvicorn")
api_key = os.environ.get('GEMINI_API_KEY')


class IngestRequest(BaseModel):
    """Solicitud para indexar documentos en Qdrant.
    
    Atributos:
        recreate: Si es True, recrea la colección (borra datos existentes).
                 Si es False, añade a la colección existente.
    """
    recreate: bool = Field(default=True, description="Si es True, recrea la colección antes de indexar")


class AskRequest(BaseModel):
    """Solicitud para hacer una pregunta al RAG.
    
    Atributos:
        question: Pregunta sobre phishing (debe tener al menos 1 carácter).
    """
    question: str = Field(min_length=1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestor de contexto del ciclo de vida de la aplicación.
    
    En el inicio:
    - Intenta hacer auto-ingesta de documentos si está configurado
    - Registra eventos de inicio y parada
    
    Yields:
        Control de la aplicación.
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
async def read_root():
    return {"message": "Phishing RAG API is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(request: IngestRequest) -> dict[str, str | int | list[str]]:
    """Indexar documentos PDFs en Qdrant.
    
    Carga documentos, genera embeddings con Google Gemini,
    y los almacena en Qdrant para posteriores búsquedas de similitud.
    
    Args:
        request: IngestRequest con bandera de recreación.
        
    Returns:
        Diccionario con estadísticas: nombre de colección, cantidad de chunks,
        archivos indexados y URLs de fuentes.
        
    Raises:
        HTTPException: Si falla la ingesta o conexión a Qdrant.
    """
    try:
        return rag_service.ingest(recreate=request.recreate)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingest failed: {exc}") from exc


@app.post("/ask")
async def ask(request: AskRequest) -> dict[str, str | list[str]]:
    """Responder una pregunta usando el RAG (Generación Aumentada con Recuperación).
    
    Busca documentos similares en el índice, construye contexto,
    y genera una respuesta con Google Gemini adjuntando las fuentes.
    
    Args:
        request: AskRequest con la pregunta sobre phishing.
        
    Returns:
        Diccionario con:
        - 'answer': Respuesta generada con fuentes adjuntas
        - 'sources': Lista de URLs de fuentes usadas
        
    Raises:
        HTTPException: Si falla la consulta o colección no existe.
    """
    try:
        return rag_service.ask(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ask failed: {exc}") from exc