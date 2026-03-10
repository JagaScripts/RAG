from fastapi import APIRouter

from src.services.vector_store import qdrant_llama_index
from src.api.schema import RAGRequest
from src.processes.llamaindex_query_engine.query_engine import router_query_engine 

router = APIRouter()

@router.post("/search")
async def search(query: str):
    retriever = qdrant_llama_index.as_retriever(similarity_top_k=5)
    found_docs = await retriever.aretrieve(query)
    return found_docs

@router.post("/rag")
async def rag_endpoint(request: RAGRequest):
    """
    Endpoint to interact with the RAG system.
    """
    response = await router_query_engine.aquery(request.question)

    return {"response": str(response), "metadata": response.metadata}