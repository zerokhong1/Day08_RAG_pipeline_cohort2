"""
Task 9 - Complete Retrieval Pipeline with Fallback.

Combines semantic search + lexical search + RRF reranking + PageIndex fallback
into a unified pipeline.

Flow:
    Query
      ├─> Semantic Search (dense, Weaviate near_vector)  -> dense_results
      ├─> Lexical Search  (BM25, rank-bm25 local)        -> sparse_results
      |
      ├─> RRF merge (Reciprocal Rank Fusion)             -> merged_results
      |
      └─> If best_score < score_threshold:
              └─> PageIndex Vectorless Fallback          -> fallback_results
"""

from __future__ import annotations

from .task5_semantic_search      import semantic_search
from .task6_lexical_search       import lexical_search
from .task7_reranking            import rerank_rrf
from .task8_pageindex_vectorless import pageindex_search


# =============================================================================
# CONFIGURATION
# =============================================================================

SCORE_THRESHOLD  = 0.3   # If best hybrid score < threshold -> PageIndex fallback
DEFAULT_TOP_K    = 5
FETCH_MULTIPLIER = 3     # Over-fetch then trim: top_k * FETCH_MULTIPLIER


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = SCORE_THRESHOLD,
    use_reranking: bool = True,
    use_pageindex_fallback: bool = True,
) -> list[dict]:
    """
    Full retrieval pipeline with hybrid search and fallback logic.

    Pipeline:
        Query
          ├─> Semantic Search    -> dense_results
          ├─> Lexical Search     -> sparse_results
          |
          ├─> RRF merge          -> merged (source = 'hybrid')
          |
          └─> if best_score < threshold:
                 └─> PageIndex   -> final_results (source = 'pageindex')

    Args:
        query:                  Search query string
        top_k:                  Number of final results to return
        score_threshold:        Minimum score threshold for hybrid results
        use_reranking:          Whether to apply RRF reranking
        use_pageindex_fallback: Whether to use PageIndex as fallback

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': str    # 'hybrid' or 'pageindex'
        }
        Sorted by score descending, length <= top_k.
    """
    fetch_k = top_k * FETCH_MULTIPLIER

    # Step 1: Run semantic + lexical search
    dense_results  = semantic_search(query, top_k=fetch_k)
    sparse_results = lexical_search(query,  top_k=fetch_k)

    # Step 2: Merge with RRF
    if use_reranking and (dense_results or sparse_results):
        ranked_lists = [r for r in [dense_results, sparse_results] if r]
        merged = rerank_rrf(ranked_lists, top_k=fetch_k)
    else:
        # Simple dedup merge without reranking
        seen: dict[str, dict] = {}
        for item in dense_results + sparse_results:
            key = item["content"]
            if key not in seen or item["score"] > seen[key]["score"]:
                seen[key] = item
        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    # Tag as hybrid source
    final_results: list[dict] = []
    for item in merged[:top_k]:
        r = item.copy()
        r["source"] = "hybrid"
        final_results.append(r)

    # Step 3: Check threshold -> PageIndex fallback
    best_score = final_results[0]["score"] if final_results else 0.0

    if use_pageindex_fallback and (not final_results or best_score < score_threshold):
        print(
            f"  [retrieve] Hybrid best_score={best_score:.3f} < "
            f"threshold={score_threshold:.3f} -> PageIndex fallback"
        )
        fallback = pageindex_search(query, top_k=top_k)
        if fallback:
            return fallback[:top_k]

        print("  [retrieve] PageIndex also returned no results")

    return final_results[:top_k]


if __name__ == "__main__":
    print("=== Task 9: Retrieval Pipeline Test ===\n")

    test_queries = [
        "Penalties for illegal drug possession under Vietnamese law",
        "Vietnamese artists arrested for drug use",
        "Drug rehabilitation mandatory procedures 2021",
        "xyzabc123nonsense",   # Nonsense query -> test fallback behavior
    ]

    for q in test_queries:
        print(f"\nQuery: {q}")
        print("-" * 60)
        results = retrieve(q, top_k=3)
        if results:
            for i, r in enumerate(results, 1):
                src = r.get("source", "?")
                print(f"  {i}. [{r['score']:.4f}] [{src}] {r['content'][:80]}...")
        else:
            print("  (No results)")
