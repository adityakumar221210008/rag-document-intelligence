from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from pydantic import BaseModel

from app.ingestion.loader import parse_document
from app.ingestion.chunker import chunk_text
from app.ingestion.embedder import embed_texts, embed_query
from app.retrieval.vector_store import VectorStore
from app.generation.llm import generate_answer
from app.memory.session import SessionMemory

router = APIRouter()

# Singletons shared across requests
store = VectorStore()
memory = SessionMemory()


# ── Upload ──────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload and index a document")
async def upload_document(file: UploadFile = File(...)):
    try:
        text = await parse_document(file)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if not text.strip():
        raise HTTPException(status_code=422, detail="Document appears to be empty.")

    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    metadata = [{"text": c, "source": file.filename} for c in chunks]
    store.add(vectors, metadata)

    return {
        "filename": file.filename,
        "chunks_indexed": len(chunks),
        "total_in_index": store.total_chunks,
    }


# ── Query ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    session_id: str = "default"
    top_k: int = 5


@router.post("/query", summary="Ask a question over indexed documents")
async def query_documents(req: QueryRequest):
    if store.total_chunks == 0:
        raise HTTPException(status_code=400, detail="No documents indexed yet. Upload a file first.")

    q_vec = embed_query(req.question)
    results = store.search(q_vec, k=req.top_k)

    context = "\n\n".join(r["chunk"]["text"] for r in results)
    sources = list({r["chunk"]["source"] for r in results})
    history = memory.get(req.session_id)

    answer = generate_answer(req.question, context, history)
    memory.add(req.session_id, req.question, answer)

    return {
        "answer": answer,
        "sources": sources,
        "retrieved_chunks": [
            {"text": r["chunk"]["text"][:200] + "...", "score": round(r["score"], 4)}
            for r in results
        ],
    }


# ── Sessions ─────────────────────────────────────────────────────────────────

@router.get("/sessions", summary="List all active sessions")
def list_sessions():
    return {"sessions": memory.list_sessions()}


@router.delete("/sessions/{session_id}", summary="Clear a session's memory")
def clear_session(session_id: str):
    memory.clear(session_id)
    return {"cleared": session_id}


# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/health")
def health():
    return {"status": "ok", "indexed_chunks": store.total_chunks}
