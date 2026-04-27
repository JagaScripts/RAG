from src.services.rag_service import rag_service


if __name__ == "__main__":
    result = rag_service.ingest(recreate=True)
    print(result)