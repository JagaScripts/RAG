\"\"\"RAG Service for Phishing Knowledge Base.\n\nCore module providing:
- Document ingestion from PDFs into Qdrant vector store\n- Semantic search and retrieval using LangChain\n- Question answering using Google Gemini LLM\n- Source attribution for all answers\n\nIntegrations:\n- LangChain for text splitting, embeddings, and LLM orchestration\n- Google Gemini API for embeddings and chat\n- Qdrant for vector similarity search\n\"\"\"\n\nfrom __future__ import annotations\n\nfrom dataclasses import dataclass\nfrom difflib import SequenceMatcher\nimport os\nfrom pathlib import Path\nimport re\nimport unicodedata\n\nfrom dotenv import load_dotenv\nfrom langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader\nfrom langchain_core.documents import Document\nfrom langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings\nfrom langchain_qdrant import QdrantVectorStore\nfrom langchain_text_splitters import RecursiveCharacterTextSplitter\nfrom qdrant_client import QdrantClient\nfrom qdrant_client.http.exceptions import ResponseHandlingException\n\n\nload_dotenv()\n\n\ndef _as_bool(value: str | None, default: bool) -> bool:\n    \"\"\"Convert environment variable string to boolean.\n    \n    Recognizes: '1', 'true', 'yes', 'y', 'on' (case-insensitive).\n    \n    Args:\n        value: Environment variable value (string or None).\n        default: Default boolean value if value is None.\n        \n    Returns:\n        Parsed boolean value.\n    \"\"\"\n    if value is None:\n        return default\n    return value.strip().lower() in {\"1\", \"true\", \"yes\", \"y\", \"on\"}\n\n\ndef _normalize(value: str) -> str:\n    \"\"\"Normalize text for URL matching.\n    \n    Performs:\n    - Lowercase normalization\n    - Fixes common 'phising' typo to 'phishing'\n    - Unicode decomposition and ASCII conversion\n    - Removes non-alphanumeric characters (keeps spaces)\n    \n    Used for matching PDF names to URLs in knowledge base file.\n    \n    Args:\n        value: Text to normalize (typically PDF name).\n        \n    Returns:\n        Normalized text suitable for fuzzy matching.\n    \"\"\"\n    lowered = value.casefold().replace(\"phising\", \"phishing\")\n    normalized = unicodedata.normalize(\"NFKD\", lowered)\n    ascii_text = normalized.encode(\"ascii\", \"ignore\").decode(\"ascii\")\n    compact = re.sub(r\"[^a-z0-9]+\", \" \", ascii_text)\n    return re.sub(r\"\\s+\", \" \", compact).strip()


@dataclass
class Settings:
    """Configuration for RAG system loaded from environment variables.
    
    Attributes:
        google_api_key: API key for Google Generative AI.
        qdrant_url: URL of Qdrant vector database.
        collection_name: Name of Qdrant collection for phishing documents.
        data_dir: Directory containing source PDF files.
        chunks_dir: Directory with pre-chunked documents (overrides PDF loading if present).
        urls_file: Path to file mapping document names to source URLs.
        similarity_top_k: Number of documents to retrieve for each query.
        chunk_size: Text chunk size for recursive splitting (characters).
        chunk_overlap: Overlap between chunks to preserve context (characters).
        auto_ingest_on_startup: Whether to index documents when service starts.
        recreate_on_startup: Whether to rebuild collection instead of appending.
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
    """Main service orchestrating document ingestion and Q&A over phishing knowledge base.
    
    Handles:
    - Configuration management from environment variables
    - PDF or pre-chunked document loading
    - Integration with Google Gemini for embeddings and LLM
    - Vector store operations with Qdrant
    - Source URL attribution for answers
    """
    
    def __init__(self) -> None:
        """Initialize RAG service with settings from environment.
        
        Loads configuration, initializes Qdrant client, and prepares
        lazy-loaded embeddings and LLM models.
        """
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
        """Verify Qdrant server is reachable before operations.
        
        Raises:
            ValueError: If Qdrant connection fails, with troubleshooting hint.
        """
        try:
            self.qdrant_client.get_collections()
        except ResponseHandlingException as exc:
            raise ValueError(
                "Cannot connect to Qdrant. Start it first with "
                "`docker compose -f docker_compose.yml up -d qdrant` and verify QDRANT_URL."
            ) from exc

    def _ensure_models(self) -> None:
        """Lazy-load Google Gemini embeddings and chat models.
        
        Initializes GoogleGenerativeAIEmbeddings for doc encoding and
        ChatGoogleGenerativeAI for LLM responses. Cached after first call.
        
        Raises:
            ValueError: If GOOGLE_API_KEY environment variable is not set.
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
        """Load and cache URL mappings from knowledge base file.
        
        Parses 'URLs base conocimiento.txt' expecting format:
        - Document Name: https://url.com
        
        Returns:
            Dictionary mapping normalized document names to source URLs.
            Cached after first load for performance.
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
        """Match a PDF file to its source URL using fuzzy string matching.
        
        First tries exact normalized match, then falls back to SequenceMatcher
        for fuzzy matching with 55% similarity threshold.
        
        Args:
            pdf_path: Path object of the PDF file.
            
        Returns:
            Source URL if found, None otherwise.
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
        """Load and parse PDF files from the data directory.
        
        Recursively searches for *.pdf files, extracts pages, and enriches
        metadata with file name and matched source URL.
        
        Returns:
            List of LangChain Document objects from all PDFs.
            
        Raises:
            ValueError: If data directory doesn't exist or no PDFs found.
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

    def _load_chunk_documents(self) -> list[Document]:\n        \"\"\"Load pre-chunked documents from the chunks directory.\n        \n        Looks for pre-processed text chunks (generated by preprocessing scripts).\n        If chunks exist, they are preferred over PDF loading for faster indexing.\n        \n        Returns:\n            List of LangChain Document objects from chunk files, or empty list\n            if chunks directory doesn't exist.\n        \"\"\"\n        chunks_path = Path(self.settings.chunks_dir)\n        if not chunks_path.exists():\n            return []\n\n        loader = DirectoryLoader(\n            str(chunks_path),\n            glob="*.txt",\n            loader_cls=TextLoader,\n            loader_kwargs={"encoding": "utf-8"},\n        )\n        docs = loader.load()\n        if not docs:\n            return []\n\n        normalized_url_mapping = self._load_urls_mapping()\n        for doc in docs:\n            source_path = str((doc.metadata or {}).get("source", ""))\n            source_name = Path(source_path).stem.split("__chunk_")[0]\n            source_url = ""\n            normalized_key = _normalize(source_name)\n            if normalized_key in normalized_url_mapping:\n                source_url = normalized_url_mapping[normalized_key]\n            else:\n                best_key: str | None = None\n                best_score = 0.0\n                for key in normalized_url_mapping:\n                    score = SequenceMatcher(a=normalized_key, b=key).ratio()\n                    if score > best_score:\n                        best_score = score\n                        best_key = key\n                if best_key and best_score >= 0.55:\n                    source_url = normalized_url_mapping[best_key]\n\n            doc.metadata = {\n                **(doc.metadata or {}),\n                "file_name": source_name,\n                "file_stem": source_name,\n                "source_url": source_url,\n            }

        return docs

    def ingest(self, recreate: bool = True) -> dict[str, str | int | list[str]]:
        """Index documents into Qdrant vector store.
        
        Strategy:
        1. Prefer pre-chunked documents if they exist (faster)
        2. Fall back to loading and splitting PDFs
        3. Generate embeddings using Google Gemini
        4. Store in Qdrant with metadata
        
        Args:
            recreate: If True, recreate collection (destroys existing data).
                     If False, append to existing collection.
                     
        Returns:
            Dictionary with ingestion statistics: collection name, page/chunk counts,
            indexed files, and source URLs.
            
        Raises:
            ValueError: If no documents found or cannot connect to Qdrant.
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
        """Get or initialize QdrantVectorStore for similarity search.
        
        Lazily loads embeddings and connects to existing Qdrant collection.
        Cached after first retrieval.
        
        Returns:
            QdrantVectorStore instance connected to the configured collection.
            
        Raises:
            ValueError: If collection doesn't exist or connection fails.
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
        """Answer a question using RAG (Retrieval-Augmented Generation).
        
        Process:
        1. Retrieve most similar documents from vector store
        2. Build context from retrieved documents
        3. Generate answer using Google Gemini with the context
        4. Extract and return source URLs for attribution
        
        Args:
            question: The user's question about phishing.
            
        Returns:
            Dictionary with:
            - 'answer': Generated response with sources appended
            - 'sources': List of source URLs used in the answer
            
        Raises:
            ValueError: If question is empty or collection not found.
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


# Global singleton instance of RAGService for FastAPI to import and use in endpoints
rag_service = RAGService()
