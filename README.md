# AI YouTube Content Intelligence Agent

Extracts a YouTube video's transcript, analyzes it, and generates a structured
blog post using a LangGraph multi-agent pipeline. Also supports Q&A over the
transcript using RAG (FAISS + embeddings).

## Setup

```bash
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# edit .env and add your GOOGLE_API_KEY (get one free at https://aistudio.google.com/apikey)
```

## Run

```bash
uvicorn main:app --reload
```

Server runs at http://127.0.0.1:8000
Interactive docs at http://127.0.0.1:8000/docs

## Try it

```bash
curl -X POST http://127.0.0.1:8000/process \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"youtube_url": "https://www.youtube.com/watch?v=VIDEO_ID", "question": "What is the main topic?"}'
```

## Architecture

```
YouTube URL
    |
    v
transcript_fetcher.py  --> raw transcript text
    |
    v
text_chunker.py  --> overlapping chunks (~800 chars, 150 overlap)
    |
    v
vector_store.py  --> FAISS index (sentence-transformers embeddings)
    |
    v
agents.py (LangGraph)
    analyze --> generate_blog --> END
    |
    v
main.py (FastAPI)  --> /process, /ask endpoints
```

## Design notes
- Uses a local HuggingFace embedding model (no API key needed for embeddings)
  so the retrieval side works even without a Gemini key.
- LLM calls (analysis + blog generation) go through Gemini via
  langchain-google-genai.
- Vector stores are cached per video_id in-memory so repeated /ask calls
  don't re-embed the transcript.
