import os

from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI


load_dotenv()


def build_llamaindex_llm() -> GoogleGenAI:
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise ValueError("Missing GOOGLE_API_KEY in environment")

    return GoogleGenAI(
        model="gemini-2.5-flash-lite",
        api_key=api_key,
        temperature=0.1,
    )