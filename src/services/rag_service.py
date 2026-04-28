"""Servicio RAG centralizado para indexación y consultas de documentos de phishing.

Utiliza LangChain + Qdrant + Google Gemini para:
- Cargar y procesar documentos PDF
- Generar embeddings y almacenarlos en una base de datos vectorial
- Recuperar documentos similares a consultas
- Generar respuestas contextuales con atribución de fuentes
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from difflib import SequenceMatcher
import os
from pathlib import Path
import re
import unicodedata

from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException


load_dotenv()


def _as_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize(value: str) -> str:
    lowered = value.casefold().replace("phising", "phishing")
    normalized = unicodedata.normalize("NFKD", lowered)
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    compact = re.sub(r"[^a-z0-9]+", " ", ascii_text)
    return re.sub(r"\s+", " ", compact).strip()


@dataclass
class Settings:
    """Configuración del servicio RAG cargada desde variables de entorno.
    
    Atributos:
        google_api_key: Clave API de Google para Gemini
        qdrant_url: URL del servidor Qdrant
        collection_name: Nombre de la colección en Qdrant
        data_dir: Directorio con documentos PDF
        chunks_dir: Directorio con chunks pre-procesados
        urls_file: Archivo con mapeo de PDF a URLs
        similarity_top_k: Número de documentos similares a recuperar
        chunk_size: Tamaño de cada chunk en caracteres
        chunk_overlap: Solapamiento entre chunks
        auto_ingest_on_startup: Auto-indexar en inicio de aplicación
        recreate_on_startup: Recrear colección en inicio
    """
    google_api_key: str
    qdrant_url: str
    collection_name: str
    data_dir: str
    chunks_dir: str
    urls_file: str
    similarity_top_k: int
    chunk_size: int
    chunk_overlap: int
    auto_ingest_on_startup: bool
    recreate_on_startup: bool


class RAGService:
    """Servicio RAG para indexación y consulta de documentos sobre phishing.
    
    Gestiona:
    - Carga de configuración desde variables de entorno
    - Conexión a Qdrant
    - Inicialización perezosa de modelos (embeddings + LLM)
    - Procesamiento de documentos
    - Consultas RAG con fuentes
    """
    def __init__(self) -> None:
        """Inicializa el servicio RAG con configuración y cliente Qdrant."""
        self.settings = Settings(
            google_api_key=os.getenv("GOOGLE_API_KEY", ""),
            qdrant_url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION", "phishing_knowledge"),
            data_dir=os.getenv("DATA_DIR", "data"),
            chunks_dir=os.getenv("CHUNKS_DIR", "data/optimized_chunks"),
            urls_file=os.getenv("KNOWLEDGE_URLS_FILE", "URLs base conocimiento.txt"),
            similarity_top_k=int(os.getenv("SIMILARITY_TOP_K", "5")),
            chunk_size=int(os.getenv("CHUNK_SIZE", "1200")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200")),
            auto_ingest_on_startup=_as_bool(os.getenv("AUTO_INGEST_ON_STARTUP"), True),
            recreate_on_startup=_as_bool(os.getenv("RECREATE_ON_STARTUP"), True),
        )

        self.qdrant_client = QdrantClient(url=self.settings.qdrant_url)
        self.embeddings: GoogleGenerativeAIEmbeddings | None = None
        self.llm: ChatGoogleGenerativeAI | None = None
        self.vector_store: QdrantVectorStore | None = None
        self._url_mapping_cache: dict[str, str] | None = None

    def _ensure_qdrant_connection(self) -> None:
        """Valida que Qdrant sea accesible.
        
        Raises:
            ValueError: Si Qdrant no responde o no está disponible.
        """
        try:
            self.qdrant_client.get_collections()
        except ResponseHandlingException as exc:
            raise ValueError(
                "Cannot connect to Qdrant. Start it first with "
                "`docker compose -f docker_compose.yml up -d qdrant` and verify QDRANT_URL."
            ) from exc

    def _ensure_qdrant_connection(self) -> None:
        try:
            self.qdrant_client.get_collections()
        except ResponseHandlingException as exc:
            raise ValueError(
                "Cannot connect to Qdrant. Start it first (e.g. `docker compose -f docker_compose.yml up -d qdrant`) "
                f"and verify QDRANT_URL={self.settings.qdrant_url}."
            ) from exc

    def _ensure_models(self) -> None:
        """Inicializa modelos de embeddings y LLM (inicialización perezosa).
        
        Crea instancias de GoogleGenerativeAIEmbeddings y ChatGoogleGenerativeAI
        la primera vez que se llamam y las cachea.
        
        Raises:
            ValueError: Si GOOGLE_API_KEY no está configurada.
        """
        if not self.settings.google_api_key:
            raise ValueError("Missing GOOGLE_API_KEY in environment")

        if self.embeddings is None:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=self.settings.google_api_key,
            )

        if self.llm is None:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                google_api_key=self.settings.google_api_key,
                temperature=0.1,
            )

    def _load_urls_mapping(self) -> dict[str, str]:
        """Carga el mapeo de nombres de PDF a URLs desde archivo.
        
        Parsea 'URLs base conocimiento.txt' buscando líneas con formato:
        '- Nombre PDF: https://url'
        
        Cachea el resultado para posteriores llamadas.
        
        Returns:
            Diccionario mapeando PDF normalizado a URL.
        """
        if self._url_mapping_cache is not None:
            return self._url_mapping_cache

        mapping: dict[str, str] = {}
        urls_path = Path(self.settings.urls_file)
        if not urls_path.exists():
            self._url_mapping_cache = mapping
            return mapping

        pattern = re.compile(r"^\s*-\s*(.+?)\s*:\s*(https?://\S+)\s*$", flags=re.IGNORECASE)
        for raw_line in urls_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            match = pattern.match(line)
            if not match:
                continue
            title, url = match.group(1), match.group(2)
            mapping[_normalize(title)] = url

        self._url_mapping_cache = mapping
        return mapping

    def _match_source_url(self, pdf_path: Path) -> str | None:
        """Busca la URL de fuente para un archivo PDF.
        
        Primero intenta coincidencia exacta. Si falla, usa coincidencia difusa
        (SequenceMatcher) con umbral del 55%.
        
        Args:
            pdf_path: Ruta al archivo PDF.
            
        Returns:
            URL de la fuente, o None si no se encuentra.
        """
        urls_mapping = self._load_urls_mapping()
        if not urls_mapping:
            return None

        pdf_key = _normalize(pdf_path.stem)
        if pdf_key in urls_mapping:
            return urls_mapping[pdf_key]

        best_key: str | None = None
        best_score = 0.0
        for key in urls_mapping:
            score = SequenceMatcher(a=pdf_key, b=key).ratio()
            if score > best_score:
                best_score = score
                best_key = key

        if best_key and best_score >= 0.55:
            return urls_mapping[best_key]
        return None

    def _load_pdf_documents(self) -> list[Document]:
        """Carga todos los archivos PDF del directorio de datos.
        
        Busca recursivamente archivos *.pdf, extrae páginas,
        y enriquece metadatos con nombre de archivo y URL de fuente.
        
        Returns:
            Lista de documentos LangChain extraidos de los PDFs.
            
        Raises:
            ValueError: Si directorio no existe o no hay PDFs.
        """
        data_path = Path(self.settings.data_dir)
        if not data_path.exists():
            raise ValueError(f"Data directory does not exist: {data_path}")

        pdf_paths = sorted(data_path.rglob("*.pdf"))
        if not pdf_paths:
            raise ValueError(f"No PDF files found in {data_path}")

        all_docs: list[Document] = []
        for pdf_path in pdf_paths:
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            source_url = self._match_source_url(pdf_path)

            for page in pages:
                page.metadata = {
                    **(page.metadata or {}),
                    "file_name": pdf_path.name,
                    "file_stem": pdf_path.stem,
                    "source_url": source_url or "",
                }

            all_docs.extend(pages)

        return all_docs

    def _load_chunk_documents(self) -> list[Document]:
        """Carga documentos pre-procesados desde el directorio de chunks.
        
        Busca archivos *.txt en data/optimized_chunks/ (preferido sobre PDFs
        para indexación más rápida). Enriquece metadatos con URLs de fuentes.
        
        Returns:
            Lista de documentos desde chunks, o lista vacía si no existe directorio.
        """
        chunks_path = Path(self.settings.chunks_dir)
        if not chunks_path.exists():
            return []

        loader = DirectoryLoader(
            str(chunks_path),
            glob="*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        docs = loader.load()
        if not docs:
            return []

        normalized_url_mapping = self._load_urls_mapping()
        for doc in docs:
            source_path = str((doc.metadata or {}).get("source", ""))
            source_name = Path(source_path).stem.split("__chunk_")[0]
            source_url = ""
            normalized_key = _normalize(source_name)
            if normalized_key in normalized_url_mapping:
                source_url = normalized_url_mapping[normalized_key]
            else:
                best_key: str | None = None
                best_score = 0.0
                for key in normalized_url_mapping:
                    score = SequenceMatcher(a=normalized_key, b=key).ratio()
                    if score > best_score:
                        best_score = score
                        best_key = key
                if best_key and best_score >= 0.55:
                    source_url = normalized_url_mapping[best_key]

            doc.metadata = {
                **(doc.metadata or {}),
                "file_name": source_name,
                "file_stem": source_name,
                "source_url": source_url,
            }

        return docs

    def ingest(self, recreate: bool = True) -> dict[str, str | int | list[str]]:
        """Indexa documentos en Qdrant.
        
        Estrategia:
        1. Prefer chunks pre-procesados si existen (más rápido)
        2. Sino, carga y divide PDFs
        3. Genera embeddings con Google Gemini
        4. Almacena en Qdrant con metadatos
        
        Args:
            recreate: Si True, recrea la colección (destruye datos existentes).
                     Si False, añade a colección existente.
                     
        Returns:
            Diccionario con estadísticas: nombre colección, páginas/chunks indexados,
            archivos indexados, y URLs de fuentes.
            
        Raises:
            ValueError: Si no hay documentos o falla conexión a Qdrant.
        """
        self._ensure_models()
        self._ensure_qdrant_connection()

        chunk_documents = self._load_chunk_documents()
        if chunk_documents:
            documents = []
            chunks = chunk_documents
        else:
            documents = self._load_pdf_documents()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
            chunks = splitter.split_documents(documents)
        if not chunks:
            raise ValueError("No chunks were generated from the PDFs")

        self.vector_store = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            url=self.settings.qdrant_url,
            collection_name=self.settings.collection_name,
            force_recreate=recreate,
        )

        indexed_files = sorted({doc.metadata.get("file_name", "") for doc in chunks if doc.metadata})
        indexed_urls = sorted(
            {
                str(doc.metadata.get("source_url"))
                for doc in chunks
                if doc.metadata and doc.metadata.get("source_url")
            }
        )

        return {
            "collection_name": self.settings.collection_name,
            "indexed_pages": len(documents),
            "indexed_chunks": len(chunks),
            "indexed_files": indexed_files,
            "indexed_urls": indexed_urls,
        }

    def _get_vector_store(self) -> QdrantVectorStore:
        """Obtiene o inicializa QdrantVectorStore para búsquedas de similitud.
        
        Carga perezosamente embeddings y conecta a colección Qdrant existente.
        Cachea después de primera recuperación.
        
        Returns:
            Instancia QdrantVectorStore conectada a colección configurada.
            
        Raises:
            ValueError: Si colección no existe o falla conexión.
        """
        self._ensure_models()
        self._ensure_qdrant_connection()

        if self.vector_store is not None:
            return self.vector_store

        if not self.qdrant_client.collection_exists(self.settings.collection_name):
            raise ValueError("Collection not found. Run POST /ingest before asking questions.")

        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embeddings,
            url=self.settings.qdrant_url,
            collection_name=self.settings.collection_name,
        )
        return self.vector_store

    def ask(self, question: str) -> dict[str, str | list[str]]:
        """Responde una pregunta usando RAG (Generación Aumentada con Recuperación).
        
        Proceso:
        1. Recupera documentos más similares del índice vectorial
        2. Construye contexto con esos documentos
        3. Genera respuesta con Google Gemini usando el contexto
        4. Extrae y retorna URLs de fuentes para atribución
        
        Args:
            question: Pregunta del usuario sobre phishing.
            
        Returns:
            Diccionario con:
            - 'answer': Respuesta generada con fuentes adjuntas
            - 'sources': Lista de URLs de fuentes usadas
            
        Raises:
            ValueError: Si pregunta está vacía o colección no existe.
        """
        clean_question = question.strip()
        if not clean_question:
            raise ValueError("Question cannot be empty")

        store = self._get_vector_store()
        docs = store.similarity_search(clean_question, k=self.settings.similarity_top_k)

        if not docs:
            return {
                "answer": "No he encontrado informacion relevante en la base de conocimiento para responder esa pregunta.\n\nFuentes:\n- No disponibles",
                "sources": [],
            }

        context = "\n\n".join(doc.page_content for doc in docs)
        prompt = (
            "Eres un asistente experto en phishing. Responde en espanol usando SOLO el contexto. "
            "Si el contexto no contiene suficiente informacion, dilo explicitamente. "
            "Devuelve una unica respuesta clara y directa, sin inventar datos.\n\n"
            f"Contexto:\n{context}\n\n"
            f"Pregunta: {clean_question}"
        )

        llm_response = self.llm.invoke(prompt)
        answer_text = (llm_response.content or "").strip()

        source_urls = sorted(
            {
                str(doc.metadata.get("source_url"))
                for doc in docs
                if doc.metadata and doc.metadata.get("source_url")
            }
        )

        if source_urls:
            sources_block = "\n".join(f"- {url}" for url in source_urls)
        else:
            sources_block = "- No disponibles en URLs base conocimiento.txt"

        return {
            "answer": f"{answer_text}\n\nFuentes:\n{sources_block}",
            "sources": source_urls,
        }


# Instancia singleton global de RAGService para que FastAPI la importe y use en endpoints
rag_service = RAGService()
