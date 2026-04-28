import json
import re
from pathlib import Path

from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def split_text(text: str, chunk_size: int = 1400, overlap: int = 250) -> list[str]:
    if not text:
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end >= text_len:
            break
        start = max(end - overlap, 0)

    return chunks


def summarize_text(text: str, max_chars: int = 420) -> str:
    if not text:
        return "Documento sin texto legible."

    sentences = re.split(r"(?<=[.!?])\s+", text)
    picked: list[str] = []
    total = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if total + len(sentence) + 1 > max_chars and picked:
            break
        picked.append(sentence)
        total += len(sentence) + 1

    if not picked:
        return text[:max_chars].strip()

    return " ".join(picked).strip()


def extract_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = [clean_text(page.extract_text() or "") for page in reader.pages]
    return clean_text("\n".join(pages))


def main() -> None:
    data_dir = Path("data")
    chunks_dir = data_dir / "optimized_chunks"
    summaries_path = data_dir / "summaries.json"

    if not data_dir.exists():
        raise FileNotFoundError("Missing data directory")

    pdf_paths = sorted(p for p in data_dir.glob("*.pdf") if p.is_file())
    if not pdf_paths:
        raise ValueError("No PDF files found in data/")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    for old_chunk in chunks_dir.glob("*.txt"):
        old_chunk.unlink()

    summaries: dict[str, str] = {}
    total_chunks = 0

    for pdf_path in pdf_paths:
        source_name = pdf_path.stem
        text = extract_pdf_text(pdf_path)
        if not text:
            continue

        summaries[source_name] = summarize_text(text)
        chunks = split_text(text)

        for idx, chunk in enumerate(chunks, start=1):
            chunk_file = chunks_dir / f"{source_name}__chunk_{idx:04d}.txt"
            chunk_file.write_text(chunk, encoding="utf-8")
            total_chunks += 1

    if not summaries:
        raise ValueError("No readable text extracted from PDFs")

    summaries_path.write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(
        {
            "pdf_count": len(pdf_paths),
            "summary_count": len(summaries),
            "chunk_count": total_chunks,
            "summaries_path": str(summaries_path),
            "chunks_dir": str(chunks_dir),
        }
    )


if __name__ == "__main__":
    main()
