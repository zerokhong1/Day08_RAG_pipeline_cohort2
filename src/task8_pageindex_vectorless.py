"""
Task 8 — PageIndex / Vectorless RAG.

LỰA CHỌN: PageIndex (pageindex.ai) yêu cầu đăng ký tài khoản + API key —
không có sẵn (xem README "Chưa có key nào cả"). Vì vậy module này:

  1. Nếu có PAGEINDEX_API_KEY trong .env → dùng PageIndex SDK thật.
  2. Nếu KHÔNG có key → tự implement một bản "vectorless retrieval" tối
     giản dựa trên STRUCTURAL UNDERSTANDING (giống tinh thần PageIndex):
     thay vì nhúng (embedding) từng đoạn vào không gian vector, ta phân
     tích cấu trúc tài liệu (heading Markdown "#", "##", Điều/Khoản trong
     văn bản luật, title bài báo) thành các "node" có ngữ cảnh phân cấp,
     rồi so khớp từ khoá (lexical overlap) giữa query và tiêu đề + nội
     dung mỗi node để chọn ra node liên quan nhất — không cần vector
     store, không cần embedding model, đúng tinh thần "vectorless".

Cài đặt (khi có key thật):
    pip install pageindex
"""

import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_ARTICLE_RE = re.compile(r"^(Điều\s+\d+\..+)$", re.MULTILINE)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


# =============================================================================
# Real PageIndex (yêu cầu API key)
# =============================================================================

def upload_documents():
    """Upload toàn bộ markdown documents lên PageIndex (cần PAGEINDEX_API_KEY)."""
    if not PAGEINDEX_API_KEY:
        raise NotImplementedError(
            "Thiếu PAGEINDEX_API_KEY — đăng ký tại https://pageindex.ai/. "
            "Hệ thống sẽ tự dùng bản vectorless tự implement (xem _structural_search)."
        )

    from pageindex import PageIndex

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        content = md_file.read_text(encoding="utf-8")
        pi.upload(
            content=content,
            metadata={"filename": md_file.name, "type": md_file.parent.name},
        )
        print(f"  ✓ Uploaded: {md_file.name}")


def _pageindex_api_search(query: str, top_k: int) -> list[dict]:
    from pageindex import PageIndex

    pi = PageIndex(api_key=PAGEINDEX_API_KEY)
    results = pi.query(query=query, top_k=top_k)
    return [
        {
            "content": r.text,
            "score": r.score,
            "metadata": r.metadata,
            "source": "pageindex",
        }
        for r in results
    ]


# =============================================================================
# Bản tự implement: vectorless retrieval dựa trên structural understanding
# (dùng khi không có PAGEINDEX_API_KEY — không cần embedding/vector store)
# =============================================================================

def _build_structure_nodes() -> list[dict]:
    """
    Phân tách mỗi document thành các "node" theo cấu trúc:
      - Văn bản luật: tách theo "Điều N. ..."
      - Bài báo / khác: tách theo heading Markdown ("#", "##", ...)
      - Fallback: cả document là 1 node

    Returns:
        List of {'title': str, 'content': str, 'metadata': dict}
    """
    nodes = []
    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        text = md_file.read_text(encoding="utf-8")
        doc_type = "legal" if "legal" in md_file.parts else "news"
        metadata = {"source": md_file.name, "type": doc_type}

        matches = list(_ARTICLE_RE.finditer(text)) or list(_HEADING_RE.finditer(text))

        if not matches:
            nodes.append({"title": md_file.stem, "content": text.strip(), "metadata": metadata})
            continue

        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section = text[start:end].strip()
            title = m.group(0).strip("# ").strip()
            if section:
                nodes.append({"title": title, "content": section, "metadata": metadata})

    return nodes


def _structural_search(query: str, top_k: int) -> list[dict]:
    """
    Lexical/structural matching: chấm điểm mỗi node bằng overlap giữa các
    token của query và token trong (title*2 trọng số + content) — ưu tiên
    node có tiêu đề khớp trực tiếp với query, mô phỏng cách PageIndex dùng
    cấu trúc tài liệu (table-of-contents / heading) để định vị thông tin
    thay vì so khớp vector.
    """
    nodes = _build_structure_nodes()
    if not nodes:
        return []

    query_tokens = Counter(_tokenize(query))
    if not query_tokens:
        return []

    scored = []
    for node in nodes:
        title_tokens = Counter(_tokenize(node["title"]))
        content_tokens = Counter(_tokenize(node["content"]))

        title_overlap = sum(min(c, title_tokens[t]) for t, c in query_tokens.items())
        content_overlap = sum(min(c, content_tokens[t]) for t, c in query_tokens.items())

        score = 2.0 * title_overlap + content_overlap
        if score > 0:
            scored.append((score, node))

    scored.sort(key=lambda x: x[0], reverse=True)

    max_score = scored[0][0] if scored else 1.0
    return [
        {
            "content": node["content"],
            "score": score / max_score,  # normalize về [0, 1]
            "metadata": node["metadata"],
            "source": "pageindex",
        }
        for score, node in scored[:top_k]
    ]


# =============================================================================
# Unified interface
# =============================================================================

def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval. Dùng PageIndex thật nếu có API key, nếu không thì
    dùng bản "structural understanding" tự implement (_structural_search) —
    cả hai đều KHÔNG dùng embedding/vector similarity.

    Returns:
        List of {'content': str, 'score': float, 'metadata': dict, 'source': 'pageindex'}
    """
    if PAGEINDEX_API_KEY:
        return _pageindex_api_search(query, top_k)
    return _structural_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Không có PAGEINDEX_API_KEY — dùng bản vectorless tự implement (structural search).")
    else:
        print("Uploading documents...")
        upload_documents()

    print("\nTest query:")
    results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
    for r in results:
        print(f"[{r['score']:.3f}] ({r['source']}) {r['content'][:100]}...")
