"""
Task 10 — Generation Có Citation.

Hướng dẫn:
    1. Chọn top_k, top_p phù hợp (giải thích lý do)
    2. Sắp xếp lại chunks sau reranking để tránh "lost in the middle"
    3. Inject context vào prompt
    4. Yêu cầu LLM trả lời có citation
    5. Nếu không đủ evidence → "Tôi không thể xác minh thông tin này"
"""

import os
from dotenv import load_dotenv

load_dotenv()

from .task9_retrieval_pipeline import retrieve


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn
# =============================================================================

# top_k: Số chunks đưa vào context
# Chọn 5 vì: đủ evidence mà không quá dài gây lost in the middle
TOP_K = 5

# top_p (nucleus sampling): Xác suất tích luỹ cho token generation
# Chọn 0.9 vì: đủ diverse nhưng không quá random
TOP_P = 0.9

# temperature: Độ ngẫu nhiên của output
# Chọn 0.3 vì: RAG cần factual, ít sáng tạo
TEMPERATURE = 0.3

# LLM provider: OpenRouter (primary) → OpenAI (fallback) → demo mode
# OpenRouter sử dụng OpenAI-compatible API, hỗ trợ nhiều model free/paid
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL    = "google/gemma-4-31b-it:free"   # confirmed free
OPENAI_MODEL        = "gpt-4o-mini"


def _get_llm_client():
    """
    Tạo OpenAI-compatible client theo thứ tự ưu tiên:
        1. OpenRouter (env: OPENROUTER)   — free models available
        2. OpenAI    (env: OPENAI_API_KEY)
        3. None → demo mode

    Returns:
        (client, model_name, provider_name) hoặc (None, None, 'demo')
    """
    from openai import OpenAI

    openrouter_key = os.getenv("OPENROUTER", "")
    openai_key     = os.getenv("OPENAI_API_KEY", "")

    if openrouter_key and not openrouter_key.startswith("sk-or-xxx"):
        client = OpenAI(
            api_key=openrouter_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/Day08-RAG-Pipeline",
                "X-Title":      "Day08 RAG Pipeline",
            },
        )
        return client, OPENROUTER_MODEL, "OpenRouter"

    if openai_key and not openai_key.startswith("sk-xxx"):
        client = OpenAI(api_key=openai_key)
        return client, OPENAI_MODEL, "OpenAI"

    return None, None, "demo"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """Trả lời câu hỏi một cách toàn diện bằng tiếng Việt.
Với mỗi nhận định thực tế, hãy ghi citation ngay sau trong ngoặc vuông,
ví dụ: [Luật Phòng chống ma tuý 2021, Điều 3] hoặc [VnExpress, 2024].

Nếu thông tin không có trong context được cung cấp, hãy trả lời:
'Tôi không thể xác minh thông tin này từ nguồn hiện có.'

Quy tắc:
- Chỉ dùng thông tin từ context được cung cấp
- Mọi nhận định thực tế PHẢI có citation
- Nếu context không đủ, nói rõ điều đó
- Cấu trúc câu trả lời rõ ràng theo đoạn văn"""


# =============================================================================
# DOCUMENT REORDERING (tránh lost in the middle)
# =============================================================================

def reorder_for_llm(chunks: list[dict]) -> list[dict]:
    """
    Sắp xếp chunks để tránh "lost in the middle" effect.

    LLM nhớ tốt thông tin ở ĐẦU và CUỐI prompt, quên thông tin ở GIỮA.
    Strategy: đặt chunks quan trọng nhất ở đầu và cuối, kém quan trọng ở giữa.

    Input order (by score):  [1, 2, 3, 4, 5]
    Output order:            [1, 3, 5, 4, 2]
    (best first, worst in middle, second-best last)

    Args:
        chunks: List sorted by score descending (from retrieval)

    Returns:
        List reordered để maximize LLM attention.
    """
    if len(chunks) <= 2:
        return chunks

    # chunks chẵn (index 0,2,4,...) → đầu prompt (LLM chú ý nhiều)
    first_half  = chunks[::2]
    # chunks lẻ (index 1,3,5,...) đảo ngược → cuối prompt (LLM cũng chú ý)
    second_half = chunks[1::2][::-1]

    return first_half + second_half


# =============================================================================
# CONTEXT FORMATTING
# =============================================================================

def format_context(chunks: list[dict]) -> str:
    """
    Format chunks thành context string cho prompt.
    Mỗi chunk có label source để LLM có thể cite.

    Args:
        chunks: List of {'content': str, 'metadata': dict, 'score': float}

    Returns:
        Formatted context string với source labels.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        meta     = chunk.get("metadata", {})
        source   = meta.get("source", f"Source {i}").replace(".md", "")
        doc_type = meta.get("doc_type") or meta.get("type", "unknown")

        context_parts.append(
            f"[Tài liệu {i} | Nguồn: {source} | Loại: {doc_type}]\n"
            f"{chunk['content']}\n"
        )

    return "\n---\n".join(context_parts)


# =============================================================================
# GENERATION
# =============================================================================

def generate_with_citation(query: str, top_k: int = TOP_K) -> dict:
    """
    End-to-end RAG generation có citation.

    Pipeline:
        1. Retrieve relevant chunks (Task 9: hybrid + PageIndex fallback)
        2. Reorder để tránh lost in the middle
        3. Format context với source labels
        4. Build prompt (system + context + query)
        5. Call LLM qua OpenRouter hoặc OpenAI
        6. Return answer + sources

    Args:
        query: Câu hỏi của user
        top_k: Số chunks đưa vào context

    Returns:
        {
            'answer': str,           # Câu trả lời có citation
            'sources': list[dict],   # Các chunks đã dùng
            'retrieval_source': str  # 'hybrid' hoặc 'pageindex'
        }
    """
    # Step 1: Retrieve
    chunks = retrieve(query, top_k=top_k)

    if not chunks:
        return {
            "answer":           "Tôi không thể xác minh thông tin này từ nguồn hiện có.",
            "sources":          [],
            "retrieval_source": "none",
        }

    retrieval_source = chunks[0].get("source", "hybrid")

    # Step 2: Reorder để tránh lost in the middle
    reordered = reorder_for_llm(chunks)

    # Step 3: Format context
    context = format_context(reordered)

    # Step 4: Build prompt
    user_message = f"Context:\n{context}\n\n---\n\nCâu hỏi: {query}"

    # Step 5: Call LLM (OpenRouter → OpenAI → demo mode)
    client, model_name, provider = _get_llm_client()

    if client is None:
        # Demo mode: không có API key
        return {
            "answer": (
                "[Demo mode — cần set OPENROUTER hoặc OPENAI_API_KEY trong .env]\n\n"
                f"Đã retrieve {len(chunks)} chunks:\n\n"
                + "\n".join(
                    f"  [{i}] {c['content'][:150]}... "
                    f"[{c.get('metadata', {}).get('source', 'N/A')}]"
                    for i, c in enumerate(chunks[:3], 1)
                )
            ),
            "sources":          chunks,
            "retrieval_source": retrieval_source,
        }

    print(f"  [generate] Provider: {provider} | Model: {model_name}")

    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    answer = response.choices[0].message.content or ""

    # Step 6: Return
    return {
        "answer":           answer,
        "sources":          chunks,
        "retrieval_source": retrieval_source,
        "provider":         provider,
        "model":            model_name,
    }


if __name__ == "__main__":
    test_queries = [
        "Hình phạt cho tội tàng trữ trái phép chất ma tuý theo pháp luật Việt Nam?",
        "Những nghệ sĩ nào đã bị bắt vì liên quan tới ma tuý?",
        "Quy trình cai nghiện bắt buộc theo Luật Phòng chống ma tuý 2021?",
    ]

    for q in test_queries:
        print(f"\n{'='*70}")
        print(f"Q: {q}")
        print("=" * 70)
        result = generate_with_citation(q)
        print(f"\nA: {result['answer']}")
        print(f"\n[Nguồn: {len(result['sources'])} chunks | via {result['retrieval_source']} | {result.get('provider','demo')}]")
