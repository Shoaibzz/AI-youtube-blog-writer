"""
vector_store.py
-----------------
Embeds transcript chunks and stores them in a FAISS vector index for
semantic retrieval. Uses a local sentence-transformers model so it works
without any API key (good for demo/interview if Gemini key isn't handy).
"""

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def build_vector_store(chunks: list[str]):
    """Builds and returns an in-memory FAISS vector store from text chunks."""
    embeddings = HuggingFaceEmbeddings(model_name=_EMBEDDING_MODEL_NAME)
    metadatas = [{"chunk_id": i} for i in range(len(chunks))]
    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings, metadatas=metadatas)
    return vector_store


def retrieve_relevant_chunks(vector_store, query: str, k: int = 4):
    """Retrieves the top-k most relevant chunks for a given query."""
    results = vector_store.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


if __name__ == "__main__":
    demo_chunks = [
        "Machine learning is a subset of AI.",
        "Deep learning uses neural networks.",
        "Cooking pasta takes about 10 minutes.",
    ]
    store = build_vector_store(demo_chunks)
    print(retrieve_relevant_chunks(store, "Tell me about neural networks"))
