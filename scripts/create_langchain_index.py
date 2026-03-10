import sys

from dotenv import load_dotenv

sys.path.append(".")
load_dotenv()

import os
import time
import warnings
from uuid import uuid4

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.services.embeddings import embeddings_model_langchain

warnings.filterwarnings("ignore")

data_path = "data/optimized_chunks"
loader = DirectoryLoader(
    data_path,
    glob="*.txt",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
)
documents = loader.load()

for doc in documents:
    source_path = doc.metadata.get("source", "")
    file_name = os.path.basename(source_path)
    source_name = os.path.splitext(file_name)[0].split("__chunk_")[0]
    doc.metadata["source"] = source_name

collection_name = "langchain_index"
qdrant_url = "http://localhost:6333"

client = QdrantClient(url=qdrant_url)

try:
    client.get_collection(collection_name)
    client.delete_collection(collection_name)
except:
    pass

client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
)

# QdrantVectorStore.from_documents(
#    documents=documents,
#    embedding=embeddings_model_langchain,
#    collection_name=collection_name,
#    url=qdrant_url,
#    prefer_grpc=True,
#    force_recreate=True
# )

vector_store = QdrantVectorStore(
    client=client, collection_name=collection_name, embedding=embeddings_model_langchain
)

for document in documents:
    doc_id = str(uuid4())
    vector_store.add_documents(documents=[document], ids=[doc_id])
    time.sleep(3)
