import os
from core.embedder import createGenAiClient
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

model = os.getenv("CHAT_MODEL")

def buildPrompt(question: str, chunks: list[str]) -> str:
    context_block = "\n\n---\n\n".join(
        f"[Chunk {i+1}]:\n{chunk}"
        for i, chunk in enumerate(chunks)
    )

    return f"""
    You are a document assistant.

    Answer the user's question using ONLY the provided context.

    Rules:
    - Use the context even if the wording is different from the question
    - You may combine information across multiple chunks
    - If the context contains relevant information, you MUST answer
    - Only say "I couldn't find that information in the document" if NONE of the chunks contain relevant information
    - Be concise and direct
    - Always mention chunk numbers used (e.g., "According to Chunk 3")

    CONTEXT:
    {context_block}

    QUESTION: {question}

    ANSWER:
    """

def generateAnswer(question: str, sourceText: list[str]) -> str:
    geminiClient = createGenAiClient()
    prompt = buildPrompt(question, sourceText)

    response = geminiClient.models.generate_content(
        model= model,
        contents= prompt,
        config= types.GenerateContentConfig(
            temperature= 0.2,
            max_output_tokens= 1024
        )
    )

    return response.text
