"""
text_chunker.py
-----------------
Splits long transcript text into overlapping chunks suitable for embedding.
Overlap preserves context across chunk boundaries, which improves retrieval
quality for RAG.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_transcript(text: str, chunk_size: int = 800, chunk_overlap: int = 150):
    """
    Splits transcript text into chunks.

    chunk_size: max characters per chunk (roughly ~150-200 tokens)
    chunk_overlap: characters shared between consecutive chunks to preserve context
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return chunks


if __name__ == "__main__":
    sample = "This is a long transcript. " * 200
    chunks = chunk_transcript(sample)
    print(f"Generated {len(chunks)} chunks")
    print("First chunk:", chunks[0][:100])
