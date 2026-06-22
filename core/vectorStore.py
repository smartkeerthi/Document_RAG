import chromadb

CHROMA_PATH = "storage/chromaDB"

def createChromaClient() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=CHROMA_PATH)

def getOrCreateCollection(chromaClient: chromadb.PersistentClient, documentId: str) -> chromadb.Collection:
    return chromaClient.get_or_create_collection(
        name= documentId,
        metadata = {"hnsm:space": "cosine"}
    )

def storeChunks(collection: chromadb.Collection, chunks: list[dict], embeddings: list[list[float]]) -> None:
    
    ids = [f"chunks_{c['chunkIndex']}" for c in chunks]

    documents = [c["text"] for c in chunks] 

    metadatas = [
        {
            "chunkIndex": c["chunkIndex"],
            "pageNumber": c["pageNumber"],
            "charStart": c["charStart"],
            "charEnd": c["charEnd"]
        }
        for c in chunks
    ]

    collection.add(
        ids= ids,
        documents= documents,
        embeddings= embeddings,
        metadatas= metadatas
    )

def searchChunks(collection: chromadb.Collection, queryEmbedded: list[float], topK: int = 4) -> dict:
    results = collection.query(
        query_embeddings=[queryEmbedded],
        n_results= min(topK, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    return results

def deleteDocument(chromaClient: chromadb.PersistentClient, documentId: str) -> None:
    chromaClient.delete_collection(name=documentId)

def listDocuments(chromaClient: chromadb.PersistentClient) -> list[str]:
    return [c.name for c in chromaClient.list_collections()]

def getCollectionCounts(chromaClient: chromadb.PersistentClient, documentId: str) -> int:
    try:
        col = chromaClient.get_collection(documentId)
        return col.count()
    except Exception:
        return 0
