import uuid
import shutil
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException
from models.schemas import UploadResponse, QueryRequest, QueryResponse, ListDocumentsResponse, DeleteResponse, DocumentInfo
import os
from dotenv import load_dotenv

from core.chunker import processPdf
from core.embedder import createGenAiClient, embedBatch
from core.vectorStore import createChromaClient, getOrCreateCollection, storeChunks, deleteDocument, getCollectionCounts, listDocuments
from core.retriever import retrieveChunks
from core.generator import generateAnswer

load_dotenv()

model = os.getenv("CHAT_MODEL")

app = FastAPI(
    title="Document Q&A API",
    description="RAG-powered document question answering using Gemini",
    version="1.0.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

document_registry: dict[str, str] = {}


@app.post("/documents/upload", response_model=UploadResponse)
async def uploadDocument(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only pdf files are supported")
    
    documentId = str(uuid.uuid4())[:8]

    filePath = UPLOAD_DIR/f"{documentId}.pdf"
    with open(filePath, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        chunks = processPdf(str(filePath))
    except Exception as e:
        filePath.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {e}")
    
    if not chunks:
        raise HTTPException(status_code=422, detail="No text found in PDF")
    
    geminiClient = createGenAiClient()
    chromaClient = createChromaClient()

    allEmbeddings = []
    batchSize = 20

    for i in range(0, len(chunks), batchSize):
        batch = chunks[i:i + batchSize]
        texts = [c["text"] for c in batch]
        # embeddings = embedBatch(geminiClient, texts)
        embeddings = embedBatch(geminiClient, texts)
        allEmbeddings.extend(embeddings)
    
    # print("all embeddings",allEmbeddings)
    print("Chunks:", len(chunks))
    print("Embeddings:", len(allEmbeddings))

    collection = getOrCreateCollection(chromaClient, documentId)
    storeChunks(collection, chunks, allEmbeddings)

    document_registry[documentId] = file.filename

    return UploadResponse(
        document_id = documentId,
        filename = file.filename,
        total_chunks = len(chunks),
        message = f"Document processed successfully into {len(chunks)} chunks"
    )


@app.post("/document/query", response_model=QueryResponse)
async def queryDocument(request: QueryRequest):

    chromaClient = createChromaClient()

    try:
        col = chromaClient.get_collection(request.document_id)
        totalChunks = col.count()
    except Exception:
        raise HTTPException(status_code=404, detail="Document not found")
    
    sources = retrieveChunks(
        documentId= request.document_id,
        question= request.question,
        topK= request.top_k,
        minConfidence= request.min_confidence
    )

    if not sources:
        return QueryResponse(
            question=request.question,
            answer="I couldn't find relevant information in the document for your question.",
            sources=[],
            total_chunks_searched=totalChunks,
            model=model,
        )
    

    collection = getOrCreateCollection(chromaClient, request.document_id)
    chunkIds = [f"chunks_{s.chunk_index}" for s in sources]
    result = collection.get(ids=chunkIds, include=["documents"])
    chunkTexts = result["documents"]

    answer = generateAnswer(request.question, chunkTexts)

    return QueryResponse(
        question= request.question,
        answer= answer,
        sources= sources,
        total_chunks_searched= totalChunks,
        model= model
    )


@app.get("/health")
async def health():
    return {"status": "ok", "model": model}