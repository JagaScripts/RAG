import argparse

from src.services.rag_service import rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the phishing RAG index from CLI")
    parser.add_argument("question", help="Question to ask")
    args = parser.parse_args()

    result = rag_service.ask(args.question)
    print("Answer:\n")
    print(result["answer"])
    print("\nSources:")
    for source in result["sources"]:
        print(f"- {source}")


if __name__ == "__main__":
    main()