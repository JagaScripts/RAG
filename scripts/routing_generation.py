import sys

sys.path.append(".")
from dotenv import load_dotenv

load_dotenv()
import asyncio
import json
import os
import warnings

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from tenacity import retry, wait_fixed

from src.services.llms import llm_langchain

warnings.filterwarnings("ignore")

prompt_template = """
Eres un asistente experto en la creación de resúmenes increibles y muy cortos de documentos. 
Tu objetivo es extraer la información clave pero general y presentarla de manera concisa y facil de entender. 
Piensa en que estos resumenes pueden servirte para entender el contenido de un documento sin leerlo.

Resume el siguiente documento:

{document_text}

Resumen:
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["document_text"])

summarization_chain = prompt | llm_langchain


@retry(wait=wait_fixed(5))
async def summarize_document(file_path):

    loader = PyPDFLoader(file_path)
    document = await asyncio.to_thread(loader.load)

    document_text = "\n\n".join(page.page_content for page in document)

    summary = await summarization_chain.ainvoke({"document_text": document_text})

    return summary.content


async def main():
    results = {}
    to_parse_documnents = os.listdir("data\\optimized_chunks")

    for doc_name in to_parse_documnents:
        if doc_name.endswith(".pdf"):
            print(f"Çhecking {doc_name}")
            file_path = os.path.join("data\\optimized_chunks", doc_name)
            summary = await summarize_document(file_path)
            if summary:
                results[doc_name.replace(".pdf", "")] = summary
                print(f"Ending {doc_name}")
            # asyncio.sleep(60)

    with open("data\\summaries.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
