from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException

load_dotenv()


def _as_bool(value: str, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    google_api_key: str
    qdrant_url: str
    collection_name: str
    data_dir: str
    similarity_top_k: int
    auto_ingest_on_startup: bool
    recreate_on_startup: bool


class RAGService:
    def __init__(self) -> None:
        self.settings = Settings(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION", "phishing_knowledge"),
            data_dir=os.getenv("DATA_DIR", "data"),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
            auto_ingest_on_startup=_as_bool(os.getenv("AUTO_INGEST_ON_STARTUP"), True),
            recreate_on_startup=_as_bool(os.getenv("RECREATE_ON_STARTUP"), True),
        )

        self.qdrant_client = QdrantClient(url=self.settings.qdrant_url)
        self.embed_model: GoogleGenAIEmbedding | None = None
        self.llm: GoogleGenAI | None = None
        self._index: VectorStoreIndex | None = None

    def _ensure_qdrant_connection(self) -> None:
        try:
            self.qdrant_client.get_collections()
        except ResponseHandlingException as exc:
            raise ValueError(
                "Cannot connect to Qdrant. Start it first (e.g. `docker compose -f docker_compose.yml up -d qdrant`) "
                f"and verify QDRANT_URL={self.settings.qdrant_url}."
            ) from exc

    def _ensure_models(self) -> None:
        if not self.settings.google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")

        if self.embed_model is None:
            self.embed_model = GoogleGenAIEmbedding(
                model_name="models/gemini-embedding-001",
                api_key=self.settings.google_api_key,
                embed_batch_size=16,
            )

        if self.llm is None:
            self.llm = GoogleGenAI(
                model="gemini-2.5-flash-lite",
                api_key=self.settings.google_api_key,
                temperature=0.1,
            )

    def _build_storage_context(self) -> StorageContext:
        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.settings.collection_name,
        )
        return StorageContext.from_defaults(vector_store=vector_store)

    def _pdf_count(self) -> int:
        data_path = Path(self.settings.data_dir)
        if not data_path.exists():
            return 0
        return len(list(data_path.rglob("*.pdf")))

    def ingest(self, recreate: bool = True) -> dict[str, str | int]:
        self._ensure_models()
        self._ensure_qdrant_connection()

        data_path = Path(self.settings.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {data_path}")

        pdf_count = self._pdf_count()
        if pdf_count == 0:
            raise ValueError(f"No PDF files found in {data_path}")

        if recreate and self.qdrant_client.collection_exists(
            self.settings.collection_name
        ):
            self.qdrant_client.delete_collection(self.settings.collection_name)

        documents = SimpleDirectoryReader(
            input_dir=str(data_path),
            recursive=True,
            required_exts=[".pdf"],
            filename_as_id=True,
        ).load_data()

        storage_context = self._build_storage_context()
        self._index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=self.embed_model,
            show_progress=False,
        )

        return {
            "collection_name": self.settings.collection_name,
            "indexed_documents": len(documents),
            "data_dir": str(data_path),
        }

    def _load_index(self) -> VectorStoreIndex:
        self._ensure_models()
        self._ensure_qdrant_connection()

        if self._index is not None:
            return self._index

        if not self.qdrant_client.collection_exists(self.settings.collection_name):
            raise ValueError(
                "Collection not found. Run POST /ingest first or enable AUTO_INGEST_ON_STARTUP."
            )

        vector_store = QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.settings.collection_name,
        )
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=self.embed_model,
        )
        return self._index

    def ask(self, question: str) -> dict[str, str | list[str]]:
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty")

        index = self._load_index()
        query_engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=self.settings.similarity_top_k,
        )

        constrained_question = (
            "Responde en espanol con foco en phishing. "
            "Usa solo informacion recuperada de los PDFs indexados. "
            "Si faltan datos, dilo de forma explicita. "
            f"Fecha actual de referencia: {date.today().isoformat()}.\n\n"
            f"Pregunta: {question}"
        )

        response = query_engine.query(constrained_question)
        source_names = []
        for node in getattr(response, "source_nodes", []):
            metadata = getattr(node, "metadata", {}) or {}
            source_name = (
                metadata.get("file_name")
                or metadata.get("filename")
                or metadata.get("source")
            )
            if source_name:
                source_names.append(str(source_name))

        unique_sources = sorted(set(source_names))
        return {
            "answer": str(response),
            "sources": unique_sources,
        }


rag_service = RAGService()
