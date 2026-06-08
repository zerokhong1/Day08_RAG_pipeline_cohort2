"""
Task 7 — Reranking Module.

Implement ĐỦ 3 phương pháp reranking:
    1. Cross-encoder (Jina Reranker v2 via API) — chất lượng cao nhất
    2. RRF (Reciprocal Rank Fusion) — gộp nhiều ranker, không cần model
    3. MMR (Maximal Marginal Relevance) — tăng diversity, giảm trùng lặp

Mặc định dùng: RRF (không cần API key, hoạt động offline)
Đổi method="cross_encoder" nếu có JINA_API_KEY.

Cài đặt:
    pip install python-dotenv requests
    # JINA_API_KEY trong .env (chỉ cần cho cross_encoder)

Giải thích từng phương pháp:
    - Cross-encoder: đọc cả (query, doc) cùng lúc → hiểu ngữ cảnh đầy đủ
    - RRF: rank fusion bằng công thức 1/(k+rank) → đơn giản, hiệu quả cao
    - MMR: chọn doc vừa relevant vừa khác với docs đã chọn → giảm trùng lặp
"""

from __future__ import annotations

import os
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# CONFIGURATION
# =============================================================================

# Jina Reranker API (chỉ cần cho method="cross_encoder")
JINA_API_KEY    = os.getenv("JINA_API_KEY", "")
JINA_MODEL      = "jina-reranker-v2-base-multilingual"
JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"

# RRF smoothing constant (Cormack et al. 2009 đề xuất k=60)
RRF_K = 60

# MMR lambda: 0.0 = pure diversity, 1.0 = pure relevance
MMR_LAMBDA = 0.7

# Phương pháp mặc định khi gọi rerank()
DEFAULT_METHOD = "rrf"   # "cross_encoder" | "rrf" | "mmr"


# =============================================================================
# Method 1: Cross-Encoder Reranking (Jina Reranker v2)
# =============================================================================

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Jina Reranker v2 base multilingual (API).

    Cơ chế Cross-Encoder:
        - Bi-encoder (Task 5) encode query và document ĐỘC LẬP → nhanh nhưng mất ngữ cảnh chéo
        - Cross-encoder encode [query, document] CÙNG LÚC qua transformer
          → hiểu được mối quan hệ giữa query và từng vị trí trong document
        - Cho relevance score chính xác hơn, nhưng chậm hơn O(n) thay vì O(1)
        - jina-reranker-v2-base-multilingual: hỗ trợ 100+ ngôn ngữ, tốt cho tiếng Việt

    API flow:
        POST https://api.jina.ai/v1/rerank
        → trả về {results: [{index, relevance_score, document}, ...]}

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    import requests

    if not JINA_API_KEY:
        raise ValueError(
            "JINA_API_KEY chưa được set trong .env\n"
            "Đăng ký tại: https://jina.ai/reranker/\n"
            "Dùng method='rrf' để chạy offline."
        )

    if not candidates:
        return []

    documents = [c["content"] for c in candidates]

    response = requests.post(
        JINA_RERANK_URL,
        headers={
            "Authorization": f"Bearer {JINA_API_KEY}",
            "Content-Type":  "application/json",
        },
        json={
            "model":     JINA_MODEL,
            "query":     query,
            "documents": documents,
            "top_n":     min(top_k, len(documents)),
        },
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("results", []):
        original = candidates[item["index"]].copy()
        original["score"] = round(item["relevance_score"], 6)
        results.append(original)

    # Sort descending (Jina đã sort, nhưng đảm bảo chắc chắn)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


# =============================================================================
# Method 2: Reciprocal Rank Fusion (RRF)
# =============================================================================

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = RRF_K
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker không cần biết score tuyệt đối.

    Công thức (Cormack et al. 2009):
        RRF(d) = Σ_{r ∈ rankers}  1 / (k + rank_r(d))

        Ví dụ: doc xuất hiện ở rank 1 trong semantic, rank 2 trong BM25:
            RRF = 1/(60+1) + 1/(60+2) = 0.01639 + 0.01613 = 0.03252

    Tại sao k=60?
        - k nhỏ → penalize mạnh rank thấp
        - k=60 là giá trị empirical tốt nhất theo paper gốc
        - Giúp document vừa-tốt ở nhiều ranker thắng document rất-tốt ở 1 ranker

    Ưu điểm:
        - Không phụ thuộc vào scale của score (cosine vs BM25 khác đơn vị)
        - Robust: document xuất hiện trong nhiều ranker được boost lên
        - Không cần model bổ sung → offline, nhanh

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số kết quả trả về
        k: Smoothing constant (default=60)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}   # content → cumulative RRF score
    content_map: dict[str, dict] = {}   # content → full result dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, start=1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            # Giữ lại item có score gốc cao nhất khi trùng key
            if key not in content_map or item.get("score", 0) > content_map[key].get("score", 0):
                content_map[key] = item

    # Sort by RRF score descending
    sorted_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    results = []
    for key in sorted_keys[:top_k]:
        item = content_map[key].copy()
        item["score"] = round(rrf_scores[key], 8)
        results.append(item)

    return results


# =============================================================================
# Method 3: Maximal Marginal Relevance (MMR)
# =============================================================================

def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa 2 vectors (đã normalize sẵn → chỉ cần dot product)."""
    return sum(x * y for x, y in zip(a, b))


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = MMR_LAMBDA,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    Công thức (Carbonell & Goldstein, 1998):
        MMR(d) = λ * sim(query, d) - (1-λ) * max_{d' ∈ Selected} sim(d, d')

        - Bước 1: Chọn doc có MMR score cao nhất trong Remaining
        - Bước 2: Thêm vào Selected, loại khỏi Remaining
        - Lặp lại cho đến đủ top_k

    Ý nghĩa lambda:
        - λ = 1.0 → pure relevance (giống semantic search, có thể trùng nhiều)
        - λ = 0.7 → balance (mặc định: ưu tiên relevance nhưng vẫn có diversity)
        - λ = 0.0 → pure diversity (chọn docs khác nhau nhất)

    Ưu điểm: Giảm trùng lặp trong kết quả — hữu ích khi chunks từ cùng tài liệu

    Args:
        query_embedding: Vector embedding của query (normalized)
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
                    LƯU Ý: candidates cần có key 'embedding' (list[float])
        top_k: Số kết quả trả về
        lambda_param: Trade-off giữa relevance và diversity [0.0, 1.0]

    Returns:
        List of top_k candidates selected by MMR, sorted by MMR score.
    """
    if not candidates:
        return []

    # Lọc candidates có embedding (MMR yêu cầu embeddings)
    valid = [c for c in candidates if "embedding" in c and c["embedding"]]
    if not valid:
        # Fallback: không có embedding → trả về top_k theo score gốc
        return sorted(candidates, key=lambda x: x.get("score", 0), reverse=True)[:top_k]

    selected_indices: list[int]   = []
    remaining_indices: list[int]  = list(range(len(valid)))

    mmr_scores: dict[int, float] = {}

    for _ in range(min(top_k, len(valid))):
        best_idx    = None
        best_mmr    = float("-inf")

        for idx in remaining_indices:
            # Relevance: cosine sim với query
            relevance = _cosine_sim(query_embedding, valid[idx]["embedding"])

            # Redundancy: max sim với đã chọn
            if selected_indices:
                max_sim = max(
                    _cosine_sim(valid[idx]["embedding"], valid[s]["embedding"])
                    for s in selected_indices
                )
            else:
                max_sim = 0.0

            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        mmr_scores[best_idx] = best_mmr

    results = []
    for idx in selected_indices:
        item = valid[idx].copy()
        item["score"] = round(mmr_scores[idx], 6)
        results.append(item)

    return results


# =============================================================================
# Unified rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = DEFAULT_METHOD,
) -> list[dict]:
    """
    Unified reranking interface — chọn phương pháp qua tham số `method`.

    Args:
        query:      Câu truy vấn
        candidates: Danh sách candidates từ retrieval
                    Format: [{'content': str, 'score': float, 'metadata': dict}]
        top_k:      Số kết quả sau rerank
        method:     "rrf" (mặc định) | "cross_encoder" | "mmr"

    Returns:
        List of top_k reranked candidates, sorted by score descending.
        Format giữ nguyên: [{'content': str, 'score': float, 'metadata': dict}]

    Examples:
        # RRF từ semantic + lexical (most common use case)
        merged = rerank(query, candidates, top_k=5, method="rrf")

        # Cross-encoder nếu có JINA_API_KEY
        reranked = rerank(query, candidates, top_k=5, method="cross_encoder")
    """
    if not candidates:
        return []

    if method == "rrf":
        # RRF cần nhiều ranked lists; khi chỉ có 1 list → wrap vào list
        # (trong Task 9, sẽ gọi rerank_rrf trực tiếp với [semantic_results, lexical_results])
        if isinstance(candidates[0], list):
            # candidates là list of lists
            return rerank_rrf(candidates, top_k=top_k)
        else:
            # candidates là 1 list → dùng score để tạo RRF từ 1 ranker
            return rerank_rrf([candidates], top_k=top_k)

    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    elif method == "mmr":
        # MMR cần query_embedding — embed query inline
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        q_emb = model.encode(query, normalize_embeddings=True).tolist()
        return rerank_mmr(q_emb, candidates, top_k=top_k)

    else:
        raise ValueError(
            f"Unknown rerank method: '{method}'. "
            "Chọn một trong: 'rrf', 'cross_encoder', 'mmr'"
        )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=== Task 7: Reranking Test ===\n")

    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý",       "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý tại nhà riêng",  "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ",           "score": 0.6, "metadata": {}},
        {"content": "Python là ngôn ngữ lập trình phổ biến",               "score": 0.3, "metadata": {}},
        {"content": "Luật Phòng, chống ma tuý 2021 quy định về xử phạt",  "score": 0.75, "metadata": {}},
    ]

    query = "hình phạt tàng trữ ma tuý"

    # Test RRF
    print(f"[RRF] Query: '{query}'")
    results = rerank(query, dummy_candidates, top_k=3, method="rrf")
    for r in results:
        print(f"  [{r['score']:.6f}] {r['content'][:80]}")
    print()

    # Test Cross-encoder (chỉ nếu có API key)
    if JINA_API_KEY:
        print(f"[Cross-Encoder] Query: '{query}'")
        try:
            results = rerank(query, dummy_candidates, top_k=3, method="cross_encoder")
            for r in results:
                print(f"  [{r['score']:.4f}] {r['content'][:80]}")
        except Exception as e:
            print(f"  ⚠ {e}")
    else:
        print("[Cross-Encoder] Skipped (JINA_API_KEY chưa set)")
