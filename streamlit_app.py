"""
streamlit_app.py
------------------
Streamlit UI for the AI YouTube Content Intelligence Agent.
Calls the agent pipeline modules directly (no need to run FastAPI separately).

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from dotenv import load_dotenv

from transcript_fetcher import fetch_transcript, extract_video_id
from text_chunker import chunk_transcript
from vector_store import build_vector_store
from agents import run_blog_pipeline, qa_agent

load_dotenv()

st.set_page_config(
    page_title="Content Intelligence Desk",
    page_icon="🎬",
    layout="centered",
)

# ---------- Session state ----------
if "video_id" not in st.session_state:
    st.session_state.video_id = None
if "analysis" not in st.session_state:
    st.session_state.analysis = None
if "blog_post" not in st.session_state:
    st.session_state.blog_post = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

# ---------- Header ----------
st.title("🎬 Content Intelligence Desk")
st.caption("Transcript → Analysis → Blog, powered by a LangGraph multi-agent pipeline")

# ---------- Input ----------
input_mode = st.radio(
    "Input method",
    ["YouTube URL", "Paste transcript manually"],
    horizontal=True,
    help="If YouTube's transcript API is flaky, paste a transcript directly to keep the demo working.",
)

video_url = None
manual_transcript = None

if input_mode == "YouTube URL":
    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://www.youtube.com/watch?v=...",
    )
else:
    video_url = st.text_input(
        "Video URL or ID (used only as a label, not fetched)",
        placeholder="https://www.youtube.com/watch?v=...",
    )
    manual_transcript = st.text_area(
        "Paste transcript text",
        height=200,
        placeholder="Paste the full transcript text here...",
    )

process_clicked = st.button("Process video", type="primary")

if process_clicked:
    if input_mode == "YouTube URL" and not video_url.strip():
        st.error("Enter a YouTube URL first.")
    elif input_mode == "Paste transcript manually" and not (manual_transcript and manual_transcript.strip()):
        st.error("Paste a transcript first.")
    else:
        try:
            if input_mode == "YouTube URL":
                with st.spinner("Fetching transcript..."):
                    video_id = extract_video_id(video_url)
                    transcript = fetch_transcript(video_url)
            else:
                with st.spinner("Using pasted transcript..."):
                    try:
                        video_id = extract_video_id(video_url) if video_url.strip() else "manual"
                    except ValueError:
                        video_id = "manual"
                    transcript = manual_transcript.strip()

            with st.spinner("Chunking transcript and building vector index..."):
                chunks = chunk_transcript(transcript)
                vector_store = build_vector_store(chunks)

            with st.spinner("Running analysis + blog generation agents..."):
                result = run_blog_pipeline(transcript)

            st.session_state.video_id = video_id
            st.session_state.analysis = result["analysis"]
            st.session_state.blog_post = result["blog_post"]
            st.session_state.vector_store = vector_store
            st.session_state.qa_history = []

            st.success(f"Done. Processed video `{video_id}` — {len(chunks)} chunks indexed.")

        except Exception as e:
            st.error(f"Something went wrong: {e}")

# ---------- Output: Analysis ----------
if st.session_state.analysis:
    st.divider()
    st.subheader("📊 Analysis")
    st.markdown(st.session_state.analysis)

# ---------- Output: Blog post ----------
if st.session_state.blog_post:
    st.divider()
    st.subheader("📝 Generated blog post")
    st.markdown(st.session_state.blog_post)
    st.download_button(
        "Download blog post (.md)",
        data=str(st.session_state.blog_post),
        file_name=f"blog_{st.session_state.video_id}.md",
        mime="text/markdown",
    )

# ---------- Output: Q&A ----------
if st.session_state.vector_store:
    st.divider()
    st.subheader("💬 Ask the video")

    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="What does this video say about...?")
        ask_clicked = st.form_submit_button("Ask")

    if ask_clicked:
        if not question or not question.strip():
            st.warning("Type a question before submitting.")
        else:
            with st.spinner("Retrieving relevant context and generating answer..."):
                try:
                    answer = qa_agent(question, st.session_state.vector_store)
                    if not answer or not str(answer).strip():
                        answer = "(The model returned an empty response — try rephrasing the question.)"
                    st.session_state.qa_history.insert(0, {"question": question, "answer": answer})
                except Exception as e:
                    st.error(f"Something went wrong: {e}")
                    st.exception(e)  # full traceback, remove once debugged

    for item in st.session_state.qa_history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            st.write(item["answer"])

st.divider()
st.caption("FastAPI backend also available — run `python -m uvicorn main:app --reload` separately if needed.")