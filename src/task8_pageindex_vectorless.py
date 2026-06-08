"""
Task 8 — PageIndex Vectorless RAG.

PageIndex là một vectorless retrieval engine: thay vì dùng embedding vectors,
nó hiểu cấu trúc tài liệu (headings, sections, bullets) để trả về đúng đoạn
văn liên quan đến query.

SDK: https://github.com/VectifyAI/PageIndex
API docs: https://pageindex.ai/docs

Cài đặt:
    pip install pageindex

Cấu hình:
    PAGEINDEX_API_KEY trong file .env

Workflow:
    1. upload_documents()   — tải các file markdown lên PageIndex (làm 1 lần)
    2. pageindex_search()   — query API để lấy kết quả (mỗi lần retrieve)
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR  = Path(__file__).parent.parent / "data" / "standardized"

# Cache document IDs sau khi upload để tránh upload lại
_CACHE_FILE = Path(__file__).parent.parent / "data" / ".pageindex_cache.json"


def _load_cache() -> dict:
    if _CACHE_FILE.exists():
        return json.loads(_CACHE_FILE.read_text(encoding="utf-8"))
    return {}


def _save_cache(cache: dict):
    _CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def upload_documents(force: bool = False) -> dict:
    """
    Upload toàn bộ markdown documents lên PageIndex.

    Mỗi file được upload một lần và document_id được cache locally.
    Gọi với force=True để re-upload tất cả.

    Args:
        force: Nếu True, upload lại dù đã upload trước đó

    Returns:
        Dict mapping filename → document_id
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("PAGEINDEX_API_KEY chưa được set trong .env")

    import requests

    cache = {} if force else _load_cache()
    headers = {
        "Authorization": f"Bearer {PAGEINDEX_API_KEY}",
        "Content-Type":  "application/json",
    }

    md_files = sorted(STANDARDIZED_DIR.rglob("*.md"))
    if not md_files:
        print("  ⚠ Không có file markdown. Hãy chạy Task 3 trước.")
        return cache

    for md_file in md_files:
        if md_file.name.startswith("."):
            continue

        filename = md_file.name
        if filename in cache and not force:
            print(f"  ↷ Skip (đã upload): {filename}")
            continue

        content  = md_file.read_text(encoding="utf-8")
        parts    = md_file.parts
        doc_type = "legal" if "legal" in parts else ("news" if "news" in parts else "unknown")

        try:
            # PageIndex API: POST /v1/documents
            resp = requests.post(
                "https://api.pageindex.ai/v1/documents",
                headers=headers,
                json={
                    "content":  content,
                    "metadata": {
                        "filename": filename,
                        "type":     doc_type,
                    },
                    "title": filename.replace(".md", "").replace("-", " "),
                },
                timeout=60,
            )
            resp.raise_for_status()
            doc_id = resp.json().get("id") or resp.json().get("document_id")
            cache[filename] = doc_id
            print(f"  ✓ Uploaded: {filename} → {doc_id}")
            time.sleep(0.5)  # Rate limiting

        except requests.HTTPError as e:
            print(f"  ✗ Upload thất bại [{filename}]: {e.response.status_code} {e.response.text[:100]}")
        except Exception as e:
            print(f"  ✗ Upload thất bại [{filename}]: {e}")

    _save_cache(cache)
    print(f"\n  ✓ Cache lưu tại: {_CACHE_FILE}")
    return cache


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.

    Thay vì tính cosine similarity giữa embedding vectors, PageIndex
    phân tích cấu trúc tài liệu (sections, paragraphs, tables) và
    tìm đoạn văn bản liên quan nhất đến query về mặt ngữ nghĩa + cấu trúc.

    Dùng làm fallback khi hybrid search không có kết quả tốt (score < threshold).

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
        Trả về [] nếu API không khả dụng hoặc chưa có API key.
    """
    if not PAGEINDEX_API_KEY:
        print("[pageindex_search] ⚠ PAGEINDEX_API_KEY chưa set. Bỏ qua PageIndex fallback.")
        return []

    import requests

    try:
        resp = requests.post(
            "https://api.pageindex.ai/v1/search",
            headers={
                "Authorization": f"Bearer {PAGEINDEX_API_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "query":   query,
                "top_k":   top_k,
                "filters": {},
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        results = []
        # PageIndex trả về: {"results": [{"text": ..., "score": ..., "metadata": ...}, ...]}
        raw_results = data.get("results") or data.get("data") or []

        for item in raw_results[:top_k]:
            content  = item.get("text") or item.get("content") or item.get("chunk", "")
            score    = float(item.get("score", 0.5))
            metadata = item.get("metadata", {})

            if not content:
                continue

            results.append({
                "content":  content,
                "score":    round(score, 6),
                "metadata": metadata,
                "source":   "pageindex",
            })

        return results

    except requests.HTTPError as e:
        status = e.response.status_code if e.response else "?"
        print(f"[pageindex_search] ⚠ HTTP {status}: {e}")
        return []
    except Exception as e:
        print(f"[pageindex_search] ⚠ Lỗi: {type(e).__name__}: {e}")
        return []


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("=== Task 8: PageIndex Vectorless RAG ===\n")

        print("[1] Uploading documents ...")
        doc_ids = upload_documents()
        print(f"    {len(doc_ids)} documents in cache\n")

        print("[2] Test search:")
        queries = [
            "hình phạt sử dụng ma tuý",
            "cai nghiện bắt buộc",
        ]
        for q in queries:
            print(f"\n  Query: '{q}'")
            results = pageindex_search(q, top_k=3)
            for i, r in enumerate(results, 1):
                print(f"    [{i}] score={r['score']:.4f} | {r['content'][:100]}...")
