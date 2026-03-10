import json
from pathlib import Path


summaries_path = Path("data") / "summaries.json"
if not summaries_path.exists():
    raise FileNotFoundError(
        "Missing data/summaries.json. Run `python -m scripts.prepare_data` before starting the API."
    )

with summaries_path.open("r", encoding="utf-8") as f:
    summaries = json.load(f)

summaries["none"] = (
    "Usa esta clasificacion para preguntas fuera del alcance de los documentos de phishing."
)

__all__ = ['summaries']