"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)

Tokenization: tiếng Việt không có khoảng trắng giữa âm tiết ghép thành từ,
nhưng việc tách theo âm tiết (lower + split theo khoảng trắng/dấu câu) vẫn
cho BM25 hoạt động hợp lý vì các âm tiết riêng lẻ ("ma", "tuý", "tàng",
"trữ"...) đã mang nhiều thông tin — không cần phụ thuộc thư viện NLP nặng
như underthesea.
"""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from .task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []  # List of {'content': str, 'metadata': dict}

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize đơn giản: lowercase + tách theo từ/âm tiết (regex \\w+)."""
    return _TOKEN_RE.findall(text.lower())


def build_bm25_index(corpus: list[dict]) -> BM25Okapi:
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}

    Returns:
        BM25Okapi index đã được fit trên corpus đã tokenize.
    """
    tokenized_corpus = [_tokenize(doc["content"]) for doc in corpus]
    return BM25Okapi(tokenized_corpus)


@lru_cache(maxsize=1)
def _get_corpus_and_index():
    """Load + chunk toàn bộ documents, build BM25 index 1 lần (cache)."""
    global CORPUS
    docs = load_documents()
    chunks = chunk_documents(docs)
    CORPUS = [{"content": c["content"], "metadata": c["metadata"]} for c in chunks]
    bm25 = build_bm25_index(CORPUS)
    return CORPUS, bm25


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    corpus, bm25 = _get_corpus_and_index()
    if not corpus:
        return []

    tokenized_query = _tokenize(query)
    scores = bm25.get_scores(tokenized_query)

    ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    results = []
    for idx in ranked_indices[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "content": corpus[idx]["content"],
            "score": float(scores[idx]),
            "metadata": corpus[idx]["metadata"],
        })
    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
