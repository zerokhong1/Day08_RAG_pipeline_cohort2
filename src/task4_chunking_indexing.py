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
"""

import sys
from pathlib import Path

# Reconfigure stdout/stderr to support Vietnamese Unicode printing on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# Chọn chunking strategy và giải thích vì sao
CHUNK_SIZE = 500        # Giải thích: Kích thước 500 chars đủ để chứa một đoạn văn có ngữ cảnh hoàn chỉnh (khoảng 100-150 từ), giúp model embedding bắt được ý chính mà không bị loãng thông tin.
CHUNK_OVERLAP = 50      # Giải thích: Overlap 50 chars giúp giữ lại sự liên kết giữa các chunk liền kề, tránh việc cắt ngang một câu hoặc một ý quan trọng.
CHUNKING_METHOD = "recursive"  # Dùng RecursiveCharacterTextSplitter vì nó chia tách văn bản một cách thông minh dựa trên cấu trúc tự nhiên (đoạn, câu, từ).

# Chọn embedding model và giải thích
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # Giải thích: Model này rất nhẹ và tải cực kỳ nhanh (~80MB), phù hợp cho việc chạy prototype mượt mà không phải chờ đợi download quá lâu.
EMBEDDING_DIM = 384

# Chọn vector store
VECTOR_STORE = "chromadb"  # Giải thích: ChromaDB là một local vector store dễ cài đặt, không cần phải chạy docker hay server riêng (như Weaviate), rất phù hợp để làm prototype hoặc lab assignment.


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
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in str(md_file) else "news"
        documents.append({
            "content": content,
            "metadata": {"source": md_file.name, "type": doc_type}
        })
    return documents


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {**doc["metadata"], "chunk_index": i}
            })
    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.
    """
    from sentence_transformers import SentenceTransformer
    
    # Lazy load model
    print(f"Đang tải embedding model {EMBEDDING_MODEL}...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    texts = [c["content"] for c in chunks]
    print("Đang embedding các chunks...")
    embeddings = model.encode(texts, show_progress_bar=True)
    
    for chunk, emb in zip(chunks, embeddings):
        chunk["embedding"] = emb.tolist()
    return chunks


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    import chromadb
    from pathlib import Path
    
    # Sử dụng ChromaDB lưu trữ local tại thư mục data/chroma_db
    db_path = Path(__file__).parent.parent / "data" / "chroma_db"
    db_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Đang kết nối tới ChromaDB tại {db_path}...")
    client = chromadb.PersistentClient(path=str(db_path))
    
    # Xoá collection cũ nếu tồn tại để tránh dữ liệu rác/trùng lặp từ các lần chạy trước
    try:
        client.delete_collection(name="law_and_news_docs")
        print("✓ Đã xoá collection cũ 'law_and_news_docs'")
    except Exception:
        pass
        
    # Tạo collection mới
    collection = client.get_or_create_collection(name="law_and_news_docs")
    
    # Chuẩn bị dữ liệu insert
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [c["embedding"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    print(f"Đang index {len(chunks)} chunks vào ChromaDB...")
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )
    print("✓ Đã index thành công vào ChromaDB!")


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
