import os

from dotenv import load_dotenv
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding


load_dotenv()


def build_llamaindex_embeddings() -> GoogleGenAIEmbedding:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in environment")

    return GoogleGenAIEmbedding(
        model_name="models/gemini-embedding-001",
        api_key=api_key,
        embed_batch_size=16,
    )
