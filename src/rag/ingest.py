from pathlib import Path
from src.config import CORPUS_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, INDEX_DIR
import numpy as np
import faiss
import pickle
from openai import OpenAI
client = OpenAI()


def load_documents():
    """Load all markdown files from the corpus directory."""
    corpus_path = Path(CORPUS_DIR)
    file_contents = []
    for fname in corpus_path.glob("*.md"):
        with open(fname, "r") as f:
            content = f.read()
            file_contents.append({"content": content, "source": fname.name})
    return file_contents


def chunk_documents(file_contents):
    if CHUNK_SIZE < CHUNK_OVERLAP:
        raise ValueError("Chunk size must be greater than chunk overlap")
    chunks = []
    for file in file_contents:
        content = file["content"]
        source = file["source"]
        chunks.extend(chunk_content(
            content, source))
    return chunks


def chunk_content(content, source):
    chunks = []
    for i in range(0, len(content), CHUNK_SIZE - CHUNK_OVERLAP):
        chunk = content[i:i + CHUNK_SIZE]
        chunks.append({"content": chunk, "source": source})
    return chunks


def embed_chunks(chunks):
    for chunk in chunks:
        embedding = client.embeddings.create(
            input=chunk["content"], model=EMBEDDING_MODEL).data[0].embedding
        chunk["embedding"] = embedding


def build_and_save_index(chunks):
    embeddings = [chunk["embedding"] for chunk in chunks]
    embeddings = np.array(embeddings).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(INDEX_DIR / "faiss_index"))
    with open(INDEX_DIR / "chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    print(f"Index saved to {INDEX_DIR / 'faiss_index'}")
    print(f"Chunks saved to {INDEX_DIR / 'chunks.pkl'}")


if __name__ == "__main__":
    file_contents = load_documents()
    # print(file_contents)

    # chunking
    chunks = chunk_documents(file_contents)
    print(f"Total chunks: {len(chunks)}")
    print(f"First chunk length: {len(chunks[0]['content'])}")
    print(f"Second chunk length: {len(chunks[1]['content'])}")

    # embedding
    embed_chunks(chunks)

    # building and saving index
    build_and_save_index(chunks)
