import sys

from src.services.rag_service import rag_service


def main() -> None:
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        raise ValueError("Usage: python -m scripts.routing_generation \"tu pregunta\"")

    result = rag_service.ask(question)
    print(result["answer"])


if __name__ == "__main__":
    main()