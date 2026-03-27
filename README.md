# RAG Document Intelligence System

A production-grade Retrieval-Augmented Generation (RAG) system that enables accurate question-answering over your own documents. Upload PDFs, DOCX, or TXT files and query them using natural language — powered by local LLMs via Ollama or OpenAI.

## Demo

```
POST /api/upload  →  Upload and index a document
POST /api/query   →  Ask a question, get an answer with sources
```

## Features

- **Multi-format ingestion** — PDF, DOCX, TXT support
- **Semantic search** — vector embeddings + FAISS for fast context retrieval
- **Source attribution** — every answer cites which document it came from
- **Conversational memory** — multi-turn Q&A with session-based chat history
- **Multi-document querying** — index many files, query across all of them
- **Flexible LLM backend** — use Ollama (free, local) or OpenAI
- **Flexible embedding backend** — local `all-MiniLM-L6-v2` or OpenAI `text-embedding-3-small`
- **Persistent vector index** — FAISS index saved to disk, survives restarts
- **Auto Swagger UI** — fully documented REST API at `/docs`

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Python 3.11+ |
| Vector Store | FAISS (IndexFlatIP) |
| Embeddings | sentence-transformers / OpenAI |
| LLM | Ollama (llama3) / OpenAI GPT-4o |
| Chunking | LangChain RecursiveCharacterTextSplitter |
| Document Parsing | pypdf, python-docx |
| Memory | In-memory session store (Redis-ready) |

## Project Structure

```
rag-system/
├── app/
│   ├── main.py              # FastAPI app + CORS
│   ├── api/
│   │   └── routes.py        # /upload, /query, /sessions, /health
│   ├── ingestion/
│   │   ├── loader.py        # PDF, DOCX, TXT parsers
│   │   ├── chunker.py       # Text splitting with overlap
│   │   └── embedder.py      # Embedding model wrapper
│   ├── retrieval/
│   │   └── vector_store.py  # FAISS index with persist/load
│   ├── generation/
│   │   └── llm.py           # LLM abstraction (Ollama + OpenAI)
│   └── memory/
│       └── session.py       # Conversational history per session
├── .env                     # Config (not committed)
├── requirements.txt
└── run.sh
```

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/adityakumar221210008/rag-document-intelligence.git
cd rag-document-intelligence
```

### 2. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure `.env`

```env
# Free local setup (no API key needed)
LLM_BACKEND=ollama
OLLAMA_MODEL=llama3
EMBEDDING_BACKEND=local

# OR paid OpenAI setup
# LLM_BACKEND=openai
# OPENAI_API_KEY=sk-your-key-here
# EMBEDDING_BACKEND=openai
```

### 4. Start Ollama (if using local LLM)

```bash
ollama pull llama3
ollama serve
```

### 5. Run the server

```bash
chmod +x run.sh && ./run.sh
```

API is live at `http://localhost:8000`
Swagger UI at `http://localhost:8000/docs`

## API Reference

### Upload a document

```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

```json
{
  "filename": "document.pdf",
  "chunks_indexed": 42,
  "total_in_index": 42
}
```

### Query documents

```bash
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?", "session_id": "user1", "top_k": 5}'
```

```json
{
  "answer": "The document is about...",
  "sources": ["document.pdf"],
  "retrieved_chunks": [
    { "text": "relevant chunk...", "score": 0.87 }
  ]
}
```

### Other endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/health` | Server status + indexed chunk count |
| GET | `/api/sessions` | List all active sessions |
| DELETE | `/api/sessions/{id}` | Clear a session's memory |

## How It Works

```
Document → Parse → Chunk → Embed → FAISS Index
                                        ↓
User Query → Embed → Similarity Search → Top-K Chunks
                                        ↓
                            Prompt Builder (query + context)
                                        ↓
                                    LLM → Answer + Sources
```

1. **Ingestion** — documents are parsed, split into overlapping chunks (512 tokens, 64 overlap), and embedded into dense vectors
2. **Retrieval** — the user query is embedded with the same model, then cosine similarity search finds the most relevant chunks in FAISS
3. **Generation** — top-k chunks are injected into a prompt alongside the query, and the LLM generates a grounded answer with source attribution
4. **Memory** — each session maintains a rolling window of the last 10 turns for multi-turn conversations

## Roadmap

- [ ] Re-ranking with cross-encoder (ms-marco-MiniLM)
- [ ] Redis-based persistent session memory
- [ ] Pinecone / Qdrant swap for production vector store
- [ ] Frontend UI (React)
- [ ] Docker Compose deployment
- [ ] Support for web URLs as document sources

## Author

**Aditya Kumar** — [github.com/adityakumar221210008](https://github.com/adityakumar221210008)  
B.Tech CSE, NIT Delhi
