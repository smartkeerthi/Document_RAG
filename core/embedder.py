import os
from google.genai import types, Client
from dotenv import load_dotenv

from concurrent.futures import ThreadPoolExecutor
from functools import partial

load_dotenv()

model = os.getenv("EMBEDDING_MODEL")

def createGenAiClient() -> Client:
    return Client(api_key=os.getenv("GOOGLE_API_KEY"))

def embedText(geminiclient: Client, text: str) -> list[float]:
    response = geminiclient.models.embed_content(
        model= model,
        contents=text
    )

    return response.embeddings[0].values

# def embedBatch(geminiclient: Client, texts: list[str]) -> list[list[float]]:
#     response = geminiclient.models.embed_content(
#         model= model,
#         contents= texts
#     )

#     print("Input texts:", len(texts))
#     print("Returned embeddings:", len(response.embeddings))

#     return [e.values for e in response.embeddings]

def embedBatch(geminiclient: Client, texts: list[str]) -> list[list[float]]:
    with ThreadPoolExecutor(max_workers= 10) as executor:
        return list(executor.map(partial(embedText, geminiclient), texts))
