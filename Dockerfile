FROM python:3.11-slim

WORKDIR /app

# Instalar git para la librería compartida
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copiar dependencias
COPY services/RAG/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código y la librería compartida
COPY services/RAG/src /app/src
COPY services/RAG/README.md /app/README.md
COPY services/lib-shared-kernel /services/lib-shared-kernel

# Instalar el kernel compartido
RUN pip install -e /services/lib-shared-kernel

ENV PYTHONPATH="/app"

EXPOSE 8000
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
