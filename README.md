# QA Knowledge Base(LLM Chatbot - MVP)

This repository contains a minimal, ready-to-run Retrieval-Augmented Generation (RAG) QA chatbot using OpenAI.

Features included:
- FastAPI backend with endpoints for ingesting documents, building embeddings and storing into Chroma vectorstore, and asking natural language questions (RAG flow)
- Demo data loader to index sample documents
- Simple Streamlit UI (`ui/app.py`) to interact with the chatbot
- **Conversation memory**: Maintains context across multiple questions for natural follow-up conversations
- Dockerfile and instructions for local run

Architecture:
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Application                       │
│                    (app/main.py)                             │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                  │
│  │ /ingest  │  │  /build  │  │  /query  │                  │
│  └────┬─────┘  └────┬─────┘  └────┬────┘                  │
│       │             │              │                        │
└───────┼─────────────┼──────────────┼────────────────────────┘
        │             │              │
        ▼             ▼              ▼
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  ingest.py   │ │  ingest.py   │ │    qa.py     │
│              │ │              │ │              │
│ • load_doc   │ │ • build_     │ │ • retrieve() │
│ • chunk_text │ │   collection │ │ • answer_    │
│ • ingest_    │ │              │ │   query()    │
│   file()     │ └──────┬───────┘ │ • call_      │
└──────┬───────┘        │         │   openai()   │
       │                │         └──────┬───────┘
       │                │                 │
       ▼                ▼                 ▼
┌──────────────────────────────────────────────────┐
│           embeddings.py                          │
│  • get_client()                                  │
│  • get_embeddings()                              │
└──────────────┬───────────────────────────────────┘
               │
               ▼
        ┌──────────────┐
        │  OpenAI API  │
        │  (Embeddings)│
        └──────────────┘

┌──────────────────────────────────────────────────┐
│           ChromaDB Vector Store                  │
│  (Persistent storage at chroma_db/)              │
└──────────────────────────────────────────────────┘


## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set your OpenAI API key in `.env` file:
   ```env
   OPENAI_API_KEY=sk-your-openai-key-here
   OPENAI_CHAT_MODEL=gpt-4o-mini
   ```

3. Load demo data:
   ```bash
   python demo_data/load_demo.py
   ```

4. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

5. Start the Streamlit UI (in another terminal):
   ```bash
   streamlit run ui/app.py --server.port 8501
   ```

## Features

### Conversation Memory
The chatbot maintains conversation history, allowing for natural follow-up questions:
- Ask: "What is the Visual Difference tool?"
- Follow up: "Tell me more about it" (the chatbot remembers what "it" refers to)
- The system automatically manages context to stay within token limits

### RAG Pipeline
- **Document Ingestion**: Upload PDFs/text files via `/ingest` endpoint
- **Vector Storage**: Documents are chunked, embedded, and stored in ChromaDB
- **Semantic Search**: Finds relevant document chunks using vector similarity
- **Context-Aware Answers**: LLM generates answers based on retrieved context

Get your OpenAI API key from: https://platform.openai.com/account/api-keys

## Internal Deployment

For deploying this tool internally within your organization, see the comprehensive **[DEPLOYMENT.md](DEPLOYMENT.md)** guide.

**Quick start with Docker Compose:**
```bash
# 1. Create .env file with your OpenAI API key
cp .env.example .env
# Edit .env and add OPENAI_API_KEY

# 2. Run deployment script
./deploy.sh
# Or manually:
docker-compose up -d
```

The deployment guide covers:
- Docker Compose deployment (recommended)
- Direct server deployment
- Kubernetes/container platforms
- Network configuration
- Security considerations
