"""
agents.py
----------
LangGraph pipeline with three specialized agents:

  1. TranscriptAnalysisAgent -> extracts key topics, structure, tone from transcript
  2. BlogGenerationAgent     -> writes a structured blog post from the analysis
  3. QAAgent                 -> answers user questions using retrieval over the transcript

The graph routes: analyze -> generate_blog -> END
QA is exposed separately since it's called on-demand, not as part of the
linear blog-generation flow.
"""

import os
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from vector_store import retrieve_relevant_chunks


def _as_text(content) -> str:
    """
    Normalizes LLM response content to a plain string.
    Some providers (including Gemini via langchain-google-genai) can return
    `content` as a list of content blocks instead of a plain string, e.g.
    [{"type": "text", "text": "..."}]. This flattens either shape to a string.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


def get_llm(temperature: float = 0.3):
    """
    Returns the LLM used by all agents.
    Requires GOOGLE_API_KEY env var. Swap model name / provider here if needed.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=temperature,
        google_api_key=os.environ.get("GOOGLE_API_KEY"),
    )


# ---------- Shared state passed between graph nodes ----------
class AgentState(TypedDict):
    transcript: str
    analysis: Optional[str]
    blog_post: Optional[str]


# ---------- Agent 1: Transcript Analysis ----------
def transcript_analysis_agent(state: AgentState) -> AgentState:
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(
        """You are a content analyst. Read the following video transcript and extract:
        1. The main topics covered (bullet points)
        2. The overall tone/style of the speaker
        3. Key takeaways a reader should walk away with

        Transcript:
        {transcript}

        Respond in structured bullet points."""
    )
    chain = prompt | llm
    result = chain.invoke({"transcript": state["transcript"][:8000]})  # guard against huge inputs
    state["analysis"] = _as_text(result.content)
    return state


# ---------- Agent 2: Blog Generation ----------
def blog_generation_agent(state: AgentState) -> AgentState:
    llm = get_llm(temperature=0.5)
    prompt = ChatPromptTemplate.from_template(
        """You are a professional content writer. Using the analysis below, write a
        well-structured blog post with:
        - An engaging title
        - An introduction
        - 3-5 sections with subheadings
        - A conclusion with key takeaways

        Analysis:
        {analysis}

        Write the full blog post in Markdown."""
    )
    chain = prompt | llm
    result = chain.invoke({"analysis": state["analysis"]})
    state["blog_post"] = _as_text(result.content)
    return state


# ---------- Build the LangGraph pipeline ----------
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("analyze", transcript_analysis_agent)
    graph.add_node("generate_blog", blog_generation_agent)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "generate_blog")
    graph.add_edge("generate_blog", END)

    return graph.compile()


def run_blog_pipeline(transcript: str) -> AgentState:
    """Runs the full analyze -> blog generation pipeline on a transcript."""
    app = build_graph()
    initial_state: AgentState = {"transcript": transcript, "analysis": None, "blog_post": None}
    final_state = app.invoke(initial_state)
    return final_state


# ---------- Agent 3: Q&A Agent (retrieval-augmented, called on demand) ----------
def qa_agent(question: str, vector_store) -> str:
    """Answers a question using top-k relevant transcript chunks as context."""
    relevant_chunks = retrieve_relevant_chunks(vector_store, question, k=4)
    context = "\n\n".join(relevant_chunks)

    llm = get_llm(temperature=0.2)
    prompt = ChatPromptTemplate.from_template(
        """Answer the question using ONLY the context below. If the answer isn't
        in the context, say you don't have enough information.

        Context:
        {context}

        Question: {question}

        Answer:"""
    )
    chain = prompt | llm
    result = chain.invoke({"context": context, "question": question})
    return _as_text(result.content)