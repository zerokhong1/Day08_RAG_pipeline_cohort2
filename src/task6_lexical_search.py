"""
Task 6 — Lexical Search Module (BM25 + Weaviate BM25 built-in).

═══════════════════════════════════════════════════════════════════════════════
[BONUS] Hai chiến lược BM25 được implement:

  1. rank-bm25 (local, offline):
     - Load toàn bộ corpus từ Weaviate hoặc thư mục standardized/
     - Tokenize bằng underthesea (nếu có) hoặc simple split()
     - Dùng BM25Okapi với k1=1.5, b=0.75

  2. Weaviate BM25 built-in (server-side):
     - Weaviate tự lập chỉ mục BM25 (Okapi BM25) trên trường "content"
     - Không cần download corpus về local
     - Nhanh hơn và scale tốt hơn cho dataset lớn
     - Cơ chế: Weaviate tokenize & index tại ingest time, query bằng bm25 operator

Mặc định dùng: local BM25 (rank-bm25) vì không phụ thuộc server khi test.
Để chuyển sang Weaviate built-in: đổi USE_WEAVIATE_BM25 = True

═══════════════════════════════════════════════════════════════════════════════
BM25 Okapi — Cách hoạt động:
    score(q, d) = Σ_i  IDF(q_i) * [tf(q_i, d) * (k1 + 1)] / [tf(q_i, d) + k1*(1 - b + b*|d|/avgdl)]

    Các tham số:
        k1 = 1.5   → term frequency saturation: từ xuất hiện nhiều vẫn cho điểm, nhưng bị giới hạn
        b  = 0.75  → length normalization: document dài hơn trung bình bị penalize nhẹ
        IDF(q_i)   → từ hiếm trong toàn corpus có trọng số cao hơn
        tf(q_i, d) → tần suất từ query trong document

    So sánh với TF-IDF:
        TF-IDF = tf * log(N/df) — đơn giản hơn, không có length normalization
        BM25   = có length norm + term saturation → thực tế cho kết quả tốt hơn

Cài đặt:
    pip install rank-bm25

Chạy:
    python src/task6_lexical_search.py
"""

from __future__ import annotations

from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Đường dẫn standardized/ để load corpus khi Weaviate không khả dụng
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

# Collection Weaviate (phải khớp Task 4)
COLLECTION_NAME  = "DrugLawDocs"
WEAVIATE_HOST    = "localhost"
WEAVIATE_PORT    = 8080

# Chọn backend: False = rank-bm25 (local), True = Weaviate BM25 built-in
USE_WEAVIATE_BM25 = False

# BM25 parameters
BM25_K1 = 1.5   # Term saturation: giá trị cao → term freq ảnh hưởng nhiều hơn
BM25_B  = 0.75  # Length normalization: 0=không normalize, 1=normalize hoàn toàn

# =============================================================================
# CORPUS LOADING
# =============================================================================

# Module-level cache để tránh load lại corpus mỗi lần gọi lexical_search
_corpus: list[dict] | None = None
_bm25_index = None


def _load_corpus_from_files() -> list[dict]:
    """
    Đọc corpus từ data/standardized/ (fallback khi Weaviate không có).
    Dùng RecursiveCharacterTextSplitter để chunk giống Task 4.
    """
    if not STANDARDIZED_DIR.exists():
        return []

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    corpus = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        if md_file.name.startswith("."):
            continue
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        parts = md_file.parts
        doc_type = "legal" if "legal" in parts else ("news" if "news" in parts else "unknown")

        for i, chunk_text in enumerate(splitter.split_text(content)):
            chunk_text = chunk_text.strip()
            if chunk_text:
                corpus.append({
                    "content":  chunk_text,
                    "metadata": {
                        "source":      md_file.name,
                        "doc_type":    doc_type,
                        "chunk_index": i,
                    },
                })
    return corpus


def _load_corpus_from_weaviate() -> list[dict]:
    """
    Lấy toàn bộ chunks từ Weaviate để build BM25 index local.
    Dùng cursor-based pagination để lấy tất cả objects.
    """
    import weaviate

    try:
        client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
    except Exception:
        return []  # Weaviate không chạy → fallback về files

    corpus = []
    try:
        if not client.collections.exists(COLLECTION_NAME):
            return []

        collection = client.collections.get(COLLECTION_NAME)

        # Lấy tất cả objects bằng fetch_objects với limit cao
        # Weaviate v4 hỗ trợ pagination qua cursor
        for obj in collection.iterator(
            return_properties=["content", "source", "doc_type", "chunk_index", "chunk_total"]
        ):
            corpus.append({
                "content": obj.properties.get("content", ""),
                "metadata": {
                    "source":      obj.properties.get("source", ""),
                    "doc_type":    obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "chunk_total": obj.properties.get("chunk_total", 0),
                },
            })
    finally:
        client.close()

    return corpus


def _get_corpus() -> list[dict]:
    """Lazy-load corpus với fallback strategy: Weaviate → Files."""
    global _corpus
    if _corpus is not None:
        return _corpus

    # Thử load từ Weaviate trước
    print("  [BM25] Loading corpus from Weaviate ...")
    _corpus = _load_corpus_from_weaviate()

    if not _corpus:
        print("  [BM25] Weaviate không có data, fallback về standardized/ files ...")
        _corpus = _load_corpus_from_files()

    if _corpus:
        print(f"  [BM25] Corpus loaded: {len(_corpus)} chunks")
    else:
        print("  [BM25] ⚠ Corpus rỗng. Chạy Task 3 và Task 4 trước.")

    return _corpus


# =============================================================================
# BM25 INDEX
# =============================================================================

def _tokenize(text: str) -> list[str]:
    """
    Tokenize văn bản tiếng Việt.

    Ưu tiên dùng underthesea (word segmentation tốt hơn cho tiếng Việt).
    Fallback về simple split() nếu underthesea chưa cài.

    Lý do quan trọng: tiếng Việt có nhiều từ ghép ("ma tuý", "chất cấm")
    → split() sẽ tách thành "ma", "tuý" riêng lẻ → BM25 kém chính xác hơn.
    """
    try:
        from underthesea import word_tokenize
        tokens = word_tokenize(text.lower(), format="text").split()
    except ImportError:
        # Fallback: lowercase + split
        tokens = text.lower().split()

    # Loại bỏ token quá ngắn (1 ký tự) nhưng giữ số
    return [t for t in tokens if len(t) > 1 or t.isdigit()]


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25Okapi index từ corpus.

    BM25Okapi (rank-bm25) được chọn vì:
        - Implement chuẩn BM25 Okapi — phổ biến nhất trong IR
        - k1=1.5, b=0.75 là giá trị mặc định được nghiên cứu tốt
        - Nhẹ, không cần server, phù hợp prototype

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi object (đã fit trên corpus)
    """
    from rank_bm25 import BM25Okapi

    print(f"  [BM25] Tokenizing {len(corpus)} documents ...")
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]

    print("  [BM25] Building BM25Okapi index ...")
    bm25 = BM25Okapi(tokenized_corpus, k1=BM25_K1, b=BM25_B)
    print("  [BM25] ✓ Index built")
    return bm25


def _get_bm25_index():
    """Lazy-load BM25 index."""
    global _bm25_index
    if _bm25_index is None:
        corpus = _get_corpus()
        if corpus:
            _bm25_index = build_bm25_index(corpus)
    return _bm25_index


# =============================================================================
# SEARCH IMPLEMENTATIONS
# =============================================================================

def _local_bm25_search(query: str, top_k: int) -> list[dict]:
    """
    Tìm kiếm bằng rank-bm25 (local, offline).
    Corpus được load và cache vào bộ nhớ.
    """
    import numpy as np

    corpus = _get_corpus()
    if not corpus:
        return []

    bm25 = _get_bm25_index()
    if bm25 is None:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    # Lấy top_k chỉ số có score cao nhất
    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_indices:
        score = float(scores[idx])
        if score <= 0:
            continue   # Bỏ qua document không có term match nào
        results.append({
            "content":  corpus[idx]["content"],
            "score":    round(score, 6),
            "metadata": corpus[idx]["metadata"],
        })

    return results


def _weaviate_bm25_search(query: str, top_k: int) -> list[dict]:
    """
    [BONUS] Tìm kiếm bằng Weaviate BM25 built-in (server-side).

    Cơ chế:
        - Weaviate tự build inverted index khi ingest document
        - Tokenization: lowercase + stop-word removal + stemming
        - BM25 scoring được tính trực tiếp trên server
        - Ưu điểm: scale tốt hơn, không cần download corpus về local
        - API: collection.query.bm25(query=..., limit=top_k)

    Scoring: Weaviate trả về score dạng BM25 absolute value (không normalized).
    Để so sánh được với semantic search, ta normalize: score / (score + 1).
    """
    import weaviate
    from weaviate.classes.query import MetadataQuery

    try:
        client = weaviate.connect_to_local(host=WEAVIATE_HOST, port=WEAVIATE_PORT)
    except Exception as e:
        raise ConnectionError(f"Không kết nối được Weaviate: {e}") from e

    results = []
    try:
        if not client.collections.exists(COLLECTION_NAME):
            raise RuntimeError(f"Collection '{COLLECTION_NAME}' chưa tồn tại. Chạy Task 4 trước.")

        collection = client.collections.get(COLLECTION_NAME)

        # Weaviate BM25 search
        response = collection.query.bm25(
            query=query,
            limit=top_k,
            return_properties=["content", "source", "doc_type", "chunk_index", "chunk_total"],
            return_metadata=MetadataQuery(score=True),
        )

        for obj in response.objects:
            raw_score = obj.metadata.score or 0.0
            # Normalize BM25 score: tránh unbounded values khi so sánh với cosine
            normalized_score = raw_score / (raw_score + 1.0)

            results.append({
                "content": obj.properties.get("content", ""),
                "score":   round(normalized_score, 6),
                "metadata": {
                    "source":      obj.properties.get("source", ""),
                    "doc_type":    obj.properties.get("doc_type", ""),
                    "chunk_index": obj.properties.get("chunk_index", 0),
                    "chunk_total": obj.properties.get("chunk_total", 0),
                },
            })
    finally:
        client.close()

    return results


# =============================================================================
# PUBLIC API
# =============================================================================

def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25 (Okapi BM25).

    Mặc định dùng rank-bm25 (local). Đổi USE_WEAVIATE_BM25=True để
    dùng Weaviate BM25 built-in (server-side, scale tốt hơn).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # BM25 score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    if USE_WEAVIATE_BM25:
        results = _weaviate_bm25_search(query, top_k)
    else:
        results = _local_bm25_search(query, top_k)

    # Đảm bảo sort descending (local BM25 đã sort, Weaviate cũng đã sort)
    results.sort(key=lambda x: x["score"], reverse=True)

    return results


if __name__ == "__main__":
    print("=== Task 6: Lexical Search (BM25) Test ===\n")
    print(f"Backend: {'Weaviate BM25 built-in' if USE_WEAVIATE_BM25 else 'rank-bm25 (local)'}")
    print(f"BM25 params: k1={BM25_K1}, b={BM25_B}\n")

    test_queries = [
        "Điều 248 tàng trữ trái phép chất ma tuý",
        "hình phạt tù phạt tiền ma tuý",
        "nghệ sĩ ca sĩ bị bắt giữ",
    ]

    for q in test_queries:
        print(f"Query: '{q}'")
        results = lexical_search(q, top_k=5)
        if results:
            for i, r in enumerate(results, 1):
                src = r["metadata"].get("source", "N/A")
                print(f"  [{i}] score={r['score']:.4f} | {src}")
                print(f"      {r['content'][:120]}...")
        else:
            print("  (Không có kết quả)")
        print()
