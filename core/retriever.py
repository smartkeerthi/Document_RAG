from core.embedder import createGenAiClient, embedText
from core.vectorStore import createChromaClient, searchChunks, getOrCreateCollection
from models.schemas import Source

def retrieveChunks(documentId: str, question: str, topK: int = 4, minConfidence: float = 0.3) -> list[Source]:
    geminiClient = createGenAiClient()
    chromaClient = createChromaClient()

    questionVector = embedText(geminiClient, question)

    collection = getOrCreateCollection(chromaClient, documentId)
    results = searchChunks(collection, questionVector, topK)

    sources = []

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for doc, meta, dis in zip(documents, metadatas, distances):
        similarities = 1 - (dis/2)

        if similarities < minConfidence:
            continue

        sources.append(Source(
            chunk_index= meta["chunkIndex"],
            page_number= meta["pageNumber"],
            text_preview= doc[:200] + "..." if len(doc) > 200 else doc,
            similarity_score= round(similarities, 4)
        ))
    
    sources.sort(key=lambda x: x.similarity_score, reverse= True)
    return sources