from typing import Literal

from pydantic import BaseModel, Field

from src.processes import summaries

categories_description = "\n".join(
    [f"- {key}: {value}" for key, value in summaries.items()]
)
possible_categories = list(summaries.keys())


class SourceModel(BaseModel):
    selection: Literal[tuple(possible_categories)] = Field(
        ...,
        description=f"Categoriza la pregunta del usuario en una de las siguientes categorias: \n{categories_description}",
    )
    reason: str = Field(..., description="Razones de tu seleccion")
