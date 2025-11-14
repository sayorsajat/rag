import json
import sys
from fastapi import FastAPI, Request
from pydantic import BaseModel
from contextlib import asynccontextmanager

import uvicorn
from llama_index.core import (
    VectorStoreIndex,
    Settings, PromptTemplate
)
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding



with open('cocktail_dataset.json', 'r', encoding="utf8") as f:
    data = json.load(f)


def load_and_prepare_documents():
    from llama_index.core import Document

    llama_documents = []
    for item in data:
        # Przetwarzanie listy składników
        ingredients_list = []
        ingredient_names_for_metadata = []

        for ing in item.get('ingredients', []):
            name = ing.get('name')
            if not name: continue

            ingredient_names_for_metadata.append(name)
            measure = ing.get('measure')

            # Dodajemy miarkę, jeśli istnieje
            if measure:
                ingredients_list.append(f"{name} ({measure.strip()})")
            else:
                ingredients_list.append(f"{name}")

        tags_list = item.get('tags') or []
        tags_str = ", ".join(tags_list)

        recipe_text = f"Name: {item['name']}\n"
        recipe_text += f"Category: {item['category']}\n"
        recipe_text += f"{'contains alcohol' if item['alcoholic'] == 1 else 'non-alcoholic'}\n"
        recipe_text += f"glass: {item['glass']}\n"

        if tags_str:
            recipe_text += f"tags: {tags_str}\n"

        recipe_text += f"ingredients: {', '.join(ingredients_list)}\n"
        recipe_text += f"instructions: {item['instructions']}"

        llama_documents.append(Document(text=recipe_text))

    print(f"Załadowano {len(llama_documents)} dokumentów")
    return llama_documents



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Inicjalizacja lokalnych modeli")
    try:
        Settings.llm = Ollama(model="tinyllama")
        Settings.embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        print("Ładowanie dokumentów")
        cocktail_documents = load_and_prepare_documents()

        print("Tworzenie indeksu wektorowego")
        index = VectorStoreIndex.from_documents(cocktail_documents)


        qa_template_str = (
            "You are a helpful bartender assistant. Answer the query based *only* on the provided context.\n"
            "Context:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n"
            "Query: {query_str}\n"
            "Answer: "
        )
        qa_template = PromptTemplate(qa_template_str)

        app.state.query_engine = index.as_query_engine(
            text_qa_template=qa_template,
            verbose=True
        )

        print("--- Serwer jest gotowy do przyjmowania zapytań ---")

    except Exception as e:
        print(f"\nBŁĄD KRYTYCZNY PRZY STARCIE")
        print(f"Szczegóły błędu: {e}")
        sys.exit(1)

    yield
    print("Zamykanie serwera...")

app = FastAPI(lifespan=lifespan)


class Query(BaseModel):
    text: str


# serwer

@app.post("/ask")
async def handle_ask_request(query: Query, request: Request):
    print(f"\n[Serwer] Otrzymano zapytanie: '{query.text}'")

    query_engine = request.app.state.query_engine

    response = await query_engine.aquery(query.text)

    print(f"[Serwer] Odpowiedź LlamaIndex: '{response.response}'")

    return {"response": str(response.response)}


if __name__ == "__main__":
    print("Uruchamianie serwera FastAPI na http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=120)
