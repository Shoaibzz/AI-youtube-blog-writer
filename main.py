"""
main.py
--------
FastAPI service exposing the YouTube Content Intelligence Agent.

Endpoints:
  POST /process   -> takes a YouTube URL, returns blog post + analysis
  POST /ask       -> takes a video URL + question, returns an answer using RAG

In-memory session store keeps vector stores per video_id so /ask can reuse
the index built during /process without re-embedding every time.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from transcript_fetcher import fetch_transcript, extract_video_id
from text_chunker import chunk_transcript
from vector_store import build_vector_store
from agents import run_blog_pipeline, qa_agent

load_dotenv()

app = FastAPI(title="AI YouTube Content Intelligence Agent")

# Simple in-memory cache: video_id -> vector_store
_SESSION_STORE = {}


class ProcessRequest(BaseModel):
    youtube_url: str


class ProcessResponse(BaseModel):
    video_id: str
    analysis: str
    blog_post: str


class AskRequest(BaseModel):
    youtube_url: str
    question: str


class AskResponse(BaseModel):
    answer: str


@app.post("/process", response_model=ProcessResponse)
def process_video(req: ProcessRequest):
    try:
        video_id = extract_video_id(req.youtube_url)
        transcript = fetch_transcript(req.youtube_url)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build and cache vector store for later Q&A
    chunks = chunk_transcript(transcript)
    vector_store = build_vector_store(chunks)
    _SESSION_STORE[video_id] = vector_store

    # Run the blog generation pipeline
    result = run_blog_pipeline(transcript)

    return ProcessResponse(
        video_id=video_id,
        analysis=result["analysis"],
        blog_post=result["blog_post"],
    )


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    video_id = extract_video_id(req.youtube_url)

    if video_id not in _SESSION_STORE:
        # Build it on the fly if /process wasn't called first
        transcript = fetch_transcript(req.youtube_url)
        chunks = chunk_transcript(transcript)
        _SESSION_STORE[video_id] = build_vector_store(chunks)

    answer = qa_agent(req.question, _SESSION_STORE[video_id])
    return AskResponse(answer=answer)


@app.get("/health")
def health_check():
    return {"status": "ok"}
