from src.config import INDEX_DIR, EMBEDDING_MODEL, TOP_K
import numpy as np
import faiss
import pickle
from openai import OpenAI
client = OpenAI()


def load_index():
    index = faiss.read_index(str(INDEX_DIR / "faiss_index"))
    return index


def load_chunks():
    with open(INDEX_DIR / "chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return chunks


def retrieve(query, index, chunks, k=TOP_K):
    # embed query
    embedding = client.embeddings.create(
        input=query, model=EMBEDDING_MODEL).data[0].embedding
    embedding = np.array([embedding])
    # retrieve chunks
    _, indices = index.search(embedding, k)
    indices = indices[0]  # unwrap the batch dimension
    retrieved_chunks = [chunks[int(i)] for i in indices]
    return retrieved_chunks


if __name__ == "__main__":
    index = load_index()
    chunks = load_chunks()
    print(f"Loaded index with {index.ntotal} vectors\n")

    # Test queries
    test_queries = [
        "What is the relocation allowance amount?",
        "How do I schedule an agent walkthrough?",
        "What can I use the $5,000 allowance for?",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        retrieved_chunks = retrieve(query, index, chunks, k=3)
        for i, chunk in enumerate(retrieved_chunks, 1):
            print(f"\n--- Chunk {i} (from: {chunk['source']}) ---")
            print(chunk["content"][:200] +
                  "..." if len(chunk["content"]) > 200 else chunk["content"])
