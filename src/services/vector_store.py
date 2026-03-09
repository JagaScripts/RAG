import os

from dotenv import load_dotenv
from llama_index.core import StorageContext
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient


load_dotenv()


def build_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=qdrant_url)


def build_storage_context(collection_name: str) -> StorageContext:
    vector_store = QdrantVectorStore(
        client=build_qdrant_client(),
        collection_name=collection_name,
    )
    return StorageContext.from_defaults(vector_store=vector_store)