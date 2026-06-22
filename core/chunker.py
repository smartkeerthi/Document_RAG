import fitz
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    chunkIndex: int
    pageNumber: int
    charStart: int
    charEnd: int


def extractTextFromPdf(filePath: str) -> list[dict]:
    doc = fitz.open(filePath)
    pages = []

    for pageNum, page in enumerate(doc, start=1):
        text = page.get_text().strip()

        if text:
            pages.append({"page": pageNum, "text": text})
        
    doc.close()

    return pages

def chunkText(text: str, pageNumber: int, chunkSize: int = 400, chunkOverlap: int = 80) -> list[dict]:

    charSize = chunkSize * 4
    charOverlap = chunkOverlap * 4

    chunks = []
    start = 0
    chunkIndex = 0

    while start < len(text):
        end = start + charSize

        if end < len(text):
            lastSpace = text.rfind(" ", start, end)
            if lastSpace > start:
                end = lastSpace
        
        chunkText = text[start:end].strip()

        if chunkText:
            chunks.append({
                "text": chunkText,
                "chunkIndex": chunkIndex,
                "pageNumber": pageNumber,
                "charStart": start,
                "charEnd": end
            })

            chunkIndex += 1
        
        start = end - charOverlap

    return chunks

def processPdf(filePath: str) -> list[dict]:
    pages = extractTextFromPdf(filePath)
    allChunks = []

    for page in pages:
        chunks = chunkText(page["text"], page["page"])
        allChunks.extend(chunks)

    for i, chunk in enumerate(allChunks):
        chunk["chunkIndex"] = i

    return chunks


processPdf("E:/Ai/RAG/ChatGPT-code.pdf")