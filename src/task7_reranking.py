"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

from typing import Optional


_qwen_model = None
_qwen_tokenizer = None

def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng Qwen/Qwen3-Reranker-0.6B.
    """
    global _qwen_model, _qwen_tokenizer
    
    # Lazy load model
    if _qwen_model is None or _qwen_tokenizer is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        print("Đang tải model Qwen/Qwen3-Reranker-0.6B...")
        model_name = "Qwen/Qwen3-Reranker-0.6B"
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _qwen_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        _qwen_model.eval()
        
    import torch
    sentence_pairs = [[query, c["content"]] for c in candidates]
    
    # Tokenize and compute scores
    with torch.no_grad():
        inputs = _qwen_tokenizer(
            sentence_pairs, 
            padding=True, 
            truncation=True, 
            return_tensors='pt', 
            max_length=1024
        )
        scores = _qwen_model(**inputs).logits.view(-1).tolist()
        
    reranked = []
    for i, c in enumerate(candidates):
        item = c.copy()
        item["score"] = float(scores[i])
        reranked.append(item)
    
    # Sort lại theo score mới
    reranked = sorted(reranked, key=lambda x: x["score"], reverse=True)
    return reranked[:top_k]


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.
    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))
    """
    import numpy as np

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    selected = []
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float('-inf')

        for idx in remaining:
            relevance = cosine_sim(query_embedding, candidates[idx]["embedding"])

            max_sim_to_selected = 0
            if selected:
                sims = [cosine_sim(candidates[idx]["embedding"], candidates[sel_idx]["embedding"]) for sel_idx in selected]
                max_sim_to_selected = max(sims)

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    # Cập nhật lại score thành mmr_score để tương thích chuẩn
    results = []
    for i in selected:
        item = candidates[i].copy()
        # Tính lại mmr score thực tế để hiển thị
        rel = cosine_sim(query_embedding, item["embedding"])
        max_sim = max([cosine_sim(item["embedding"], candidates[s]["embedding"]) for s in selected if s != i] + [0])
        item["score"] = lambda_param * rel - (1 - lambda_param) * max_sim
        results.append(item)
        
    return results

def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.
    RRF(d) = Σ 1 / (k + rank_r(d))
    """
    rrf_scores = {}  # content -> score
    content_map = {}  # content -> full dict

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item["content"]
            rrf_scores[key] = rrf_scores.get(key, 0) + 1.0 / (k + rank)
            content_map[key] = item

    # Sort by RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "mmr",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Embed query và candidates nếu chưa có embedding
        from sentence_transformers import SentenceTransformer
        # Dùng chung model siêu nhẹ với Task 4/5
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        query_embedding = model.encode(query).tolist()
        for c in candidates:
            if "embedding" not in c:
                c["embedding"] = model.encode(c["content"]).tolist()
                
        return rerank_mmr(query_embedding, candidates, top_k)
    elif method == "rrf":
        # RRF cần nhiều ranked lists
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data using MMR
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2, method="mmr")
    for r in results:
        print(f"[{r.get('score', 0):.3f}] {r['content']}")
