"""Punto de entrada para la API del RAG de phishing.

Inicia el servidor FastAPI usando Uvicorn en el puerto 8000.
"""
import warnings
import uvicorn

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False, env_file=".env")
