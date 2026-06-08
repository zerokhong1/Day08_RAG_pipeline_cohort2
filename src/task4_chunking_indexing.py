"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters sentence-transformers weaviate-client

Chạy Weaviate local bằng Docker:
    docker run -d -p 8080:8080 -p 50051:50051 cr.weaviate.io/semitechnologies/weaviate:latest
"""

from pathlib import Path

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Tên collection trong Weaviate
COLLECTION_NAME = "DrugLawDocs"

# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# ── Chunking Strategy: RecursiveCharacterTextSplitter ──────────────────────
# Lý do chọn:
#   • An toàn nhất cho văn bản hỗn hợp (pháp luật + tin tức).
#   • Tách theo thứ tự ưu tiên: đoạn văn → dòng → câu → từ → ký tự,
#     nên luôn giữ nguyên ý nghĩa ở mức cao nhất có thể.
#   • Không yêu cầu cấu trúc heading cố định như MarkdownHeaderTextSplitter.
CHUNK_SIZE    = 500   # 500 ký tự ≈ 3–5 câu → đủ ngữ cảnh, không bị loãng
CHUNK_OVERLAP = 50    # 50 ký tự overlap → tránh mất thông tin ở ranh giới chunk
CHUNKING_METHOD = "recursive"   # "recursive" | "markdown_header" | "semantic"

# ── Embedding Model: all-MiniLM-L6-v2 ──────────────────────────────────────
# Lý do chọn:
#   • 384 chiều — rất nhẹ, encode cả bộ data trong vài giây trên CPU.
#   • Được tối ưu cho semantic similarity, phù hợp RAG search.
#   • Chạy hoàn toàn offline, không cần API key.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM   = 384

# ── Vector Store: FAISS ──────────────────────────────────────────────────
# Lý do chọn:
#   • Đơn giản, chạy trực tiếp trên máy không cần cài đặt Docker.
#   • Rất phù hợp để chạy Semantic Search (dense retrieval).
VECTOR_STORE = "faiss"   # "weaviate" | "chromadb" | "faiss"


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    if not STANDARDIZED_DIR.exists():
        raise FileNotFoundError(
            f"Không tìm thấy thư mục: {STANDARDIZED_DIR}\n"
            "Hãy chạy Task 3 trước để chuẩn hóa dữ liệu."
        )

    documents = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):          # bỏ qua .gitkeep
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            print(f"  ⚠ Bỏ qua file rỗng: {md_file.name}")
            continue

        # Xác định loại tài liệu dựa vào thư mục cha
        parts = md_file.parts
        if "legal" in parts:
            doc_type = "legal"
        elif "news" in parts:
            doc_type = "news"
        else:
            doc_type = "unknown"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "type":   doc_type,
            }
        })
        print(f"  [OK] Loaded [{doc_type}] {md_file.name} ({len(content):,} chars)")

    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Strategy: RecursiveCharacterTextSplitter
        Tách theo thứ tự: \\n\\n → \\n → ". " → " " → ""
        Đảm bảo chunk không vượt quá CHUNK_SIZE ký tự,
        với CHUNK_OVERLAP ký tự chồng lấn giữa các chunk liền kề.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunk_total": len(splits),
                }
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Model: sentence-transformers/all-MiniLM-L6-v2
        - 384 chiều, chạy offline trên CPU
        - Batch encode để tối ưu tốc độ

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [c["content"] for c in chunks]

    print(f"  Encoding {len(texts)} chunks ...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine similarity chuẩn hơn
    )

    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()

    return chunks


def index_to_vectorstore(chunks: list[dict]) -> int:
    """
    Lưu chunks vào Vector Store đã chọn.
    Returns:
        Số lượng object đã insert thành công.
    """
    if VECTOR_STORE == "faiss":
        import os
        import pickle
        import numpy as np
        import faiss

        print("  Xây dựng FAISS index...")
        os.makedirs("data/faiss", exist_ok=True)
        
        embeddings = np.array([c["embedding"] for c in chunks], dtype=np.float32)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        
        faiss.write_index(index, "data/faiss/index.faiss")
        
        with open("data/faiss/chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
            
        print("  [OK] Đã lưu FAISS index và metadata vào data/faiss/")
        return len(chunks)

    elif VECTOR_STORE == "weaviate":
        import weaviate
        from weaviate.classes.config import Configure, Property, DataType

        print("  Kết nối Weaviate tại localhost:8080 ...")
        try:
            client = weaviate.connect_to_local()
        except Exception as e:
            raise ConnectionError(
                f"Không thể kết nối Weaviate: {e}\n"
                "Hãy chắc chắn Weaviate đang chạy:\n"
                "  docker run -d -p 8080:8080 -p 50051:50051 "
                "cr.weaviate.io/semitechnologies/weaviate:latest"
            ) from e

        try:
            if client.collections.exists(COLLECTION_NAME):
                client.collections.delete(COLLECTION_NAME)
                print(f"  ↺ Đã xóa collection cũ: {COLLECTION_NAME}")

            collection = client.collections.create(
                name=COLLECTION_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT),
                    Property(name="source", data_type=DataType.TEXT),
                    Property(name="doc_type", data_type=DataType.TEXT),
                    Property(name="chunk_index", data_type=DataType.INT),
                    Property(name="chunk_total", data_type=DataType.INT),
                ],
            )
            print(f"  [OK] Đã tạo collection: {COLLECTION_NAME}")

            failed = 0
            with collection.batch.dynamic() as batch:
                for chunk in chunks:
                    batch.add_object(
                        properties={
                            "content":     chunk["content"],
                            "source":      chunk["metadata"]["source"],
                            "doc_type":    chunk["metadata"]["type"],
                            "chunk_index": chunk["metadata"]["chunk_index"],
                            "chunk_total": chunk["metadata"]["chunk_total"],
                        },
                        vector=chunk["embedding"],
                    )
                failed = batch.number_errors

            inserted = len(chunks) - failed
            if failed > 0:
                print(f"  ⚠ {failed} chunk(s) insert thất bại")

            return inserted
        finally:
            client.close()
    else:
        raise NotImplementedError(f"Chưa hỗ trợ vector store: {VECTOR_STORE}")


def verify_indexing() -> dict:
    """
    Kiểm tra nhanh kết quả sau khi index: đếm tổng object,
    thử 1 vector search để xác nhận pipeline hoạt động.

    Returns:
        {'total': int, 'sample_result': str}
    """
    from sentence_transformers import SentenceTransformer

    if VECTOR_STORE == "faiss":
        import faiss
        import pickle
        import numpy as np
        
        index = faiss.read_index("data/faiss/index.faiss")
        with open("data/faiss/chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
            
        total = index.ntotal
        
        model  = SentenceTransformer(EMBEDDING_MODEL)
        q_vec  = model.encode(["ma túy"], normalize_embeddings=True)
        distances, indices = index.search(np.array(q_vec, dtype="float32"), 1)
        
        sample = ""
        if len(indices) > 0 and indices[0][0] != -1:
            idx = indices[0][0]
            chunk = chunks[idx]
            sample = (
                f"[{chunk['metadata']['type']}] {chunk['metadata']['source']}\n"
                f"  → {chunk['content'][:120]}..."
            )

        return {"total": total, "sample_result": sample}
    
    elif VECTOR_STORE == "weaviate":
        import weaviate

        client = weaviate.connect_to_local()
        try:
            collection = client.collections.get(COLLECTION_NAME)
            response   = collection.aggregate.over_all(total_count=True)
            total      = response.total_count

            model  = SentenceTransformer(EMBEDDING_MODEL)
            q_vec  = model.encode("ma túy", normalize_embeddings=True).tolist()
            result = collection.query.near_vector(
                near_vector=q_vec,
                limit=1,
                return_properties=["content", "source", "doc_type"],
            )

            sample = ""
            if result.objects:
                obj    = result.objects[0]
                sample = (
                    f"[{obj.properties['doc_type']}] {obj.properties['source']}\n"
                    f"  → {obj.properties['content'][:120]}..."
                )

            return {"total": total, "sample_result": sample}
        finally:
            client.close()
    return {"total": 0, "sample_result": ""}


def run_pipeline():
    """Chạy toàn bộ pipeline: load -> chunk -> embed -> index -> verify."""
    print("=" * 60)
    print("Task 4: Chunking & Indexing Pipeline")
    print("-" * 60)
    print(f"  Strategy : {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Store    : {VECTOR_STORE} / collection={COLLECTION_NAME}")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────
    print("\n[1/4] Loading documents ...")
    docs = load_documents()
    if not docs:
        print("  ✗ Không có tài liệu nào để xử lý. Hãy chạy Task 3 trước.")
        return
    print(f"  → {len(docs)} documents loaded\n")

    # ── 2. Chunk ─────────────────────────────────────────────────
    print("[2/4] Chunking documents ...")
    chunks = chunk_documents(docs)
    avg_len = sum(len(c["content"]) for c in chunks) / max(len(chunks), 1)
    print(f"  → {len(chunks)} chunks created (avg {avg_len:.0f} chars/chunk)\n")

    # ── 3. Embed ─────────────────────────────────────────────────
    print("[3/4] Embedding chunks ...")
    chunks = embed_chunks(chunks)
    print(f"  → {len(chunks)} embeddings generated (dim={EMBEDDING_DIM})\n")

    # ── 4. Index ─────────────────────────────────────────────────
    print(f"[4/4] Indexing to {VECTOR_STORE} ...")
    inserted = index_to_vectorstore(chunks)
    print(f"  → {inserted}/{len(chunks)} chunks indexed successfully\n")

    # ── 5. Verify ────────────────────────────────────────────────
    print("[5/4] Verifying ...")
    try:
        info = verify_indexing()
        print(f"  [OK] Total objects in {VECTOR_STORE}: {info['total']}")
        if info["sample_result"]:
            print(f"  Sample search ('ma túy'):\n    {info['sample_result']}")
    except Exception as e:
        print(f"  ⚠ Verify skipped: {e}")

    print("\n" + "=" * 60)
    print("✅ Task 4 hoàn thành!")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()
