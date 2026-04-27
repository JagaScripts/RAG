"""Entry point for the Phishing RAG API.

Starts the FastAPI server using Uvicorn with the configured host and port.
Reload is disabled for production-like behavior.
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=False)