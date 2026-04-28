from langchain_qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.vector_stores.qdrant import QdrantVectorStore as LlamaQdrantVectorStore
from qdrant_client import AsyncQdrantClient, QdrantClient

from src.services.embeddings import (
    embeddings_model_langchain,
    embeddings_model_llama_index,
)

### LANGCHAIN
qdrant_langchain = QdrantVectorStore.from_existing_collection(
    embedding=embeddings_model_langchain,
    collection_name="langchain_index",
    url="http://localhost:6333",
)

## QDRANT CLIENT
qdrant_client = QdrantClient(host="localhost", port=6333)
async_qdrant_client = AsyncQdrantClient(host="localhost", port=6333)


### LLAMA INDEX
vector_store = LlamaQdrantVectorStore(
    client=QdrantClient(url="http://localhost:6333"),
    aclient=AsyncQdrantClient(url="http://localhost:6333"),
    collection_name="llamaindex_index",
    batch_size=1,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)
qdrant_llama_index = VectorStoreIndex.from_documents(
    [], storage_context=storage_context, embed_model=embeddings_model_llama_index
)
