import os
from google.genai import types, Client
from dotenv import load_dotenv

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

def embedBatch(geminiclient: Client, texts: list[str]) -> list[list[float]]:
    response = geminiclient.models.embed_content(
        model= model,
        contents= texts
    )

    return [e.values for e in response.embeddings]
