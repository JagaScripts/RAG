import logging
from contextlib import asynccontextmanager
import os

from fastapi import FastAPI

# Routers
from src.api.router_clients import router as clients_rag_router
from src.api.router_langchain import router as langchain_rag_router
from src.api.router_llamaindex import router as llamaindex_rag_router

logger = logging.getLogger("uvicorn")
api_key = os.environ.get('GEMINI_API_KEY')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models and other resources
    logger.info("Starting up...")
    yield
    # Clean up the ML models and other resources
    logger.info("Shutting down...")


app = FastAPI(
    title="RAG and Semantic Search API",
    description="An API for RAG and Semantic Search with LangChain, LlamaIndex, and direct clients.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def read_root():
    return {"message": "Phishing RAG API is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


# Adding Routers
app.include_router(langchain_rag_router, prefix="/langchain", tags=["Langchain System"])
app.include_router(
    llamaindex_rag_router, prefix="/llama_index", tags=["Llama Index System"]
)
app.include_router(
    clients_rag_router, prefix="/clients", tags=["Direct Clients System"]
)
