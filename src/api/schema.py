from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    recreate: bool = Field(
        default=True,
        description="If true, the collection is recreated before indexing PDFs.",
    )


class IngestResponse(BaseModel):
    collection_name: str
    indexed_documents: int
    data_dir: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, description="Question to answer using the indexed PDFs.")


class AskResponse(BaseModel):
    answer: str
    sources: list[str]