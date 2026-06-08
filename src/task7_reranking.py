"""
Task 7 — Reranking Module.

LỰA CHỌN: tự implement MMR + RRF (không cần API key như Jina/Cohere).

  - RRF (Reciprocal Rank Fusion): dùng để GỘP kết quả từ nhiều ranker khác
    nhau (semantic + lexical) — xem Task 9. Cơ chế: mỗi document nhận điểm
    1/(k+rank) từ mỗi ranked list rồi cộng dồn — ưu điểm là không cần các
    ranker dùng cùng thang điểm (semantic score 0-1 vs BM25 score 0-30+).

  - MMR (Maximal Marginal Relevance): dùng làm phương pháp RERANK chính —
    vừa chấm lại độ liên quan với query (dựa trên cosine similarity của
    embedding) vừa giảm trùng lặp/tăng đa dạng giữa các chunk được chọn.
    Phù hợp với corpus có nhiều đoạn lặp ý (văn bản luật hay lặp định nghĩa,
    bài báo hay lặp thông tin nền) — chọn MMR giúp câu trả lời cuối cùng
    bao phủ nhiều khía cạnh hơn thay vì 5 chunk gần như giống nhau.

  MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, đã chọn))
  λ = 0.7 → ưu tiên độ liên quan (relevance) hơn diversity, nhưng vẫn đủ
  trọng số để loại bớt các chunk trùng lặp gần như hoàn toàn.
"""

from typing import Optional

import numpy as np


def _cosine_sim(a, b) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
    return float(np.dot(a, b) / denom)


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model qua Jina Reranker API.

    Lưu ý: cần JINA_API_KEY trong .env. Nếu không có, hãy dùng method="mmr"
    hoặc "rrf" (tự implement, không cần API key) — xem rerank().
    """
    import os
    import requests
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("JINA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "Thiếu JINA_API_KEY trong .env — hãy dùng method='mmr' hoặc 'rrf' "
            "(tự implement, không cần API key) thay thế."
        )

    response = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": "jina-reranker-v2-base-multilingual",
            "query": query,
            "documents": [c["content"] for c in candidates],
            "top_n": top_k,
        },
        timeout=30,
    )
    response.raise_for_status()
    reranked = response.json()["results"]
    return [
        {**candidates[r["index"]], "score": r["relevance_score"]}
        for r in reranked
    ]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR, mỗi item có 'score' = MMR relevance.
    """
    selected: list[int] = []
    remaining = list(range(len(candidates)))

    # Pre-compute relevance (sim với query) — không đổi qua các vòng lặp
    relevance = [_cosine_sim(query_embedding, c["embedding"]) for c in candidates]

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            max_sim_to_selected = 0.0
            for sel_idx in selected:
                sim = _cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"])
                max_sim_to_selected = max(max_sim_to_selected, sim)

            mmr_score = lambda_param * relevance[idx] - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected.append(best_idx)
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = {k: v for k, v in candidates[idx].items() if k != "embedding"}
        item["score"] = relevance[idx]
        results.append(item)
    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores: dict[str, float] = {}
    content_map: dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = {kk: vv for kk, vv in content_map[content].items() if kk != "embedding"}
        item["score"] = score
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "mmr" | "rrf" | "cross_encoder"
) -> list[dict]:
    """
    Unified reranking interface.

    Mặc định dùng "mmr" (tự implement, không cần API key — embed query +
    candidates bằng chính embedding model ở Task 4 rồi áp dụng MMR).

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking — "mmr" | "rrf" | "cross_encoder"

    Returns:
        List of top_k reranked candidates.
    """
    if not candidates:
        return []

    if method == "mmr":
        from .task4_chunking_indexing import get_embedding_model

        model = get_embedding_model()
        query_embedding = model.encode(query, normalize_embeddings=True).tolist()

        # Embed candidates chưa có sẵn 'embedding' (vd: kết quả từ lexical search)
        missing = [i for i, c in enumerate(candidates) if "embedding" not in c]
        if missing:
            texts = [candidates[i]["content"] for i in missing]
            embeddings = model.encode(texts, normalize_embeddings=True).tolist()
            for i, emb in zip(missing, embeddings):
                candidates[i] = {**candidates[i], "embedding": emb}

        return rerank_mmr(query_embedding, candidates, top_k=top_k)

    elif method == "rrf":
        # RRF gộp nhiều ranked lists — dùng trực tiếp rerank_rrf([...]) ở Task 9.
        # Ở đây coi candidates như 1 ranked list duy nhất (rank theo thứ tự đầu vào).
        return rerank_rrf([candidates], top_k=top_k)

    elif method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)

    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
