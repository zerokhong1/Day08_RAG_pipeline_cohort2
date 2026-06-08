"""
Task 4 — Chunking & Indexing vào Vector Store.

LỰA CHỌN & LÝ DO
================

1) Chunking: RecursiveCharacterTextSplitter (langchain-text-splitters)
   - CHUNK_SIZE = 800, CHUNK_OVERLAP = 120
   - Vì sao 800? Văn bản pháp luật tiếng Việt có các Điều/Khoản dài, 800 ký
     tự (~150-200 từ) đủ để giữ trọn 1 đơn vị ý nghĩa (1 khoản hoặc vài câu
     liên quan) mà không quá dài gây loãng embedding.
   - Vì sao overlap 120 (~15% size)? Đảm bảo câu/ý nằm ở ranh giới 2 chunk
     không bị cắt rời, giúp retrieval không bỏ sót ngữ cảnh.
   - Recursive splitter ưu tiên tách theo đoạn ("\n\n") → câu (". ") → từ,
     an toàn cho cả văn bản luật (có cấu trúc heading) lẫn bài báo (văn xuôi).

2) Embedding: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
   - 384 dimensions, hỗ trợ 50+ ngôn ngữ bao gồm tiếng Việt.
   - Nhẹ (~470MB) và đủ nhanh để chạy local trên CPU (không cần GPU/API key)
     — phù hợp với môi trường không có OpenAI/Cohere key.

3) Vector store: ChromaDB (local, persistent)
   - Không cần Docker hay tài khoản cloud (khác Weaviate).
   - Hỗ trợ lưu persistent trên đĩa (PersistentClient) + cosine similarity.
   - Đơn giản để cài đặt & demo, phù hợp cho dataset nhỏ (vài trăm chunks).

Cài đặt:
    pip install langchain-text-splitters sentence-transformers chromadb
"""

import os

# Tránh "OMP: Error #15: Initializing libiomp5md.dll, but found ... already
# initialized" — xung đột OpenMP runtime giữa torch và onnxruntime trên Windows
# có thể gây access-violation crash. Phải set TRƯỚC khi import torch/onnxruntime.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from functools import lru_cache
from pathlib import Path

# NOTE: import thứ tự này (langchain_text_splitters TRƯỚC sentence_transformers/
# torch) tránh một xung đột native DLL (OpenMP/onnxruntime) trên Windows từng
# gây access-violation crash khi load embedding model. Giữ nguyên thứ tự import.
from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: F401

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
CHROMA_DIR = Path(__file__).parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "drug_law_docs"


# =============================================================================
# CONFIGURATION
# =============================================================================

CHUNK_SIZE = 800        # Đủ giữ trọn 1 Điều/Khoản hoặc đoạn văn bản báo chí
CHUNK_OVERLAP = 120     # ~15% size — tránh cắt rời câu ở ranh giới chunk
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384  # multilingual, hỗ trợ tiếng Việt, chạy tốt trên CPU

VECTOR_STORE = "chromadb"  # local, persistent, không cần Docker/cloud


# =============================================================================
# SHARED HELPERS (dùng lại ở Task 5/6/7)
# =============================================================================

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load (và cache) embedding model — share giữa indexing & search."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_chroma_collection():
    """Lấy (hoặc tạo) ChromaDB collection persistent trên đĩa."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type},
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents bằng RecursiveCharacterTextSplitter.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng EMBEDDING_MODEL.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    model = get_embedding_model()
    texts = [c["content"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """Lưu chunks (kèm embedding) vào ChromaDB collection (ghi đè nếu đã tồn tại)."""
    import chromadb

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Xoá collection cũ để tránh trùng lặp khi chạy lại pipeline
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )
    get_chroma_collection.cache_clear()

    ids = [f"{c['metadata']['source']}::{c['metadata']['chunk_index']}" for c in chunks]
    documents = [c["content"] for c in chunks]
    embeddings = [c["embedding"] for c in chunks]
    metadatas = [
        {"source": c["metadata"]["source"], "type": c["metadata"]["type"],
         "chunk_index": c["metadata"]["chunk_index"]}
        for c in chunks
    ]

    # Insert theo batch để tránh vượt giới hạn của Chroma
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
