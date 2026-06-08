"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


def upload_documents():
    if not PAGEINDEX_API_KEY:
        raise ValueError("Lỗi: PAGEINDEX_API_KEY chưa được cấu hình. Vui lòng kiểm tra file .env!")
        
    from pageindex import PageIndexClient
    import tempfile
    
    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # Hàm phụ để convert txt/md sang PDF do PageIndex chỉ nhận PDF
    def convert_to_pdf(md_path):
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()
        # Dùng font Arial mặc định của Windows để hỗ trợ Tiếng Việt
        font_path = "C:/Windows/Fonts/arial.ttf"
        try:
            pdf.add_font("Arial", "", font_path, uni=True)
            pdf.set_font("Arial", size=12)
        except Exception:
            pdf.set_font("Helvetica", size=12)
            
        content = md_path.read_text(encoding="utf-8")
        # Ghi từng dòng bằng write để tránh lỗi wrapping của multi_cell
        for line in content.split('\n'):
            if line.strip():
                try:
                    pdf.write(8, text=line + '\n')
                except Exception:
                    pass
            else:
                pdf.ln(8)
            
        out_path = md_path.with_suffix(".pdf")
        pdf.output(str(out_path))
        return out_path
    
    for md_file in STANDARDIZED_DIR.rglob("*.md"):
        try:
            print(f"  Đang chuyển đổi {md_file.name} sang PDF...")
            pdf_file = convert_to_pdf(md_file)
            res = pi.submit_document(file_path=str(pdf_file))
            print(f"  ✓ Uploaded: {pdf_file.name} -> doc_id: {res.get('doc_id')}")
        except Exception as e:
            print(f"  ✗ Lỗi khi upload {md_file.name}: {e}")


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

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
    """
    # User đã cấu hình API Key, bắt buộc dùng hệ thống PageIndex thật.
    if not PAGEINDEX_API_KEY:
        raise ValueError("Lỗi: PAGEINDEX_API_KEY chưa được cấu hình. Vui lòng kiểm tra file .env!")
        
    from pageindex import PageIndexClient
    import time
    from concurrent.futures import ThreadPoolExecutor
    
    pi = PageIndexClient(api_key=PAGEINDEX_API_KEY)
    
    # 1. Lấy danh sách documents
    docs_resp = pi.list_documents(limit=50)
    docs = docs_resp.get("documents", [])
    
    if not docs:
        print("Không có document nào trên PageIndex. Vui lòng chạy upload_documents trước.")
        return []
        
    # 2. Kiểm tra trạng thái ready song song
    def check_ready(doc):
        doc_id = doc.get("id")
        if doc_id:
            try:
                if pi.is_retrieval_ready(doc_id):
                    return doc
            except Exception:
                pass
        return None
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        ready_docs = list(filter(None, executor.map(check_ready, docs)))
        
    if not ready_docs:
        print("Chưa có document nào sẵn sàng để truy vấn.")
        return []
        
    # 3. Submit query song song
    def submit(doc):
        try:
            res = pi.submit_query(doc_id=doc["id"], query=query)
            if "retrieval_id" in res:
                return (doc, res["retrieval_id"])
        except Exception as e:
            print(f"Lỗi submit query cho doc {doc.get('id')}: {e}")
        return None
        
    with ThreadPoolExecutor(max_workers=10) as executor:
        retrieval_jobs = list(filter(None, executor.map(submit, ready_docs)))
        
    # 4. Polling kết quả song song theo nhóm
    all_results = []
    active_jobs = list(retrieval_jobs)
    
    def poll_job(job):
        doc, r_id = job
        try:
            res = pi.get_retrieval(r_id)
            return (job, res)
        except Exception:
            return (job, None)
            
    for _ in range(5):
        if not active_jobs:
            break
            
        with ThreadPoolExecutor(max_workers=10) as executor:
            poll_results = executor.map(poll_job, active_jobs)
            
        still_running = []
        for job, res in poll_results:
            doc, r_id = job
            if res and res.get("status") == "completed":
                nodes = res.get("result", {}).get("nodes", [])
                for node in nodes:
                    all_results.append({
                        "content": node.get("content", ""),
                        "score": float(node.get("relevance_score", 0.5)),
                        "metadata": {"filename": doc.get("name", ""), "doc_id": doc.get("id")},
                        "source": "pageindex"
                    })
            else:
                still_running.append(job)
                
        active_jobs = still_running
        if active_jobs:
            time.sleep(2)
            
    # Sắp xếp kết quả theo score và trả về top_k
    all_results = sorted(all_results, key=lambda x: x["score"], reverse=True)
    return all_results[:top_k]


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
