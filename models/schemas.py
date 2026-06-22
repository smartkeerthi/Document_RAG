from pydantic import BaseModel
from typing import Optional

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    total_chunks: int
    message: str

class QueryRequest(BaseModel):
    document_id: str
    question: str
    top_k: int = 4            
    min_confidence: float = 0.3  

class Source(BaseModel):
    chunk_index: int
    page_number: Optional[int]
    text_preview: str         
    similarity_score: float

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    total_chunks_searched: int
    model: str

class DocumentInfo(BaseModel):
    document_id: str
    filename: str
    total_chunks: int

class ListDocumentsResponse(BaseModel):
    documents: list[DocumentInfo]
    total: int

class DeleteResponse(BaseModel):
    document_id: str
    message: str