import argparse

from src.services.rag_service import rag_service


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest all PDFs from the configured data directory")
    parser.add_argument(
        "--no-recreate",
        action="store_true",
        help="Append data to existing collection instead of recreating it.",
    )
    args = parser.parse_args()

    result = rag_service.ingest(recreate=not args.no_recreate)
    print(result)


if __name__ == "__main__":
    main()