"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import sys

# Reconfigure stdout/stderr to support Vietnamese Unicode printing on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
if sys.stderr.encoding != 'utf-8':
    try:
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass


def normalize_vietnamese_accents(text: str) -> str:
    if not text:
        return text
    replacements = {
        "hoà": "hòa", "hoá": "hóa", "hoả": "hỏa", "hoã": "hõa", "hoạ": "họa",
        "oà": "òa", "oá": "óa", "oả": "ỏa", "oã": "õa", "oạ": "ọa",
        "uỳ": "ùy", "uý": "úy", "uỷ": "ủy", "uỹ": "ũy", "uỵ": "ụy",
        "oè": "òe", "oé": "óe", "oẻ": "ỏe", "oẽ": "õe", "oẹ": "ọe",
        "uề": "uề", "uế": "uế", "uể": "uể", "uễ": "uễ", "uệ": "uệ",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
        text = text.replace(old.upper(), new.upper())
        text = text.replace(old.capitalize(), new.capitalize())
    return text


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    query = normalize_vietnamese_accents(query)
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    import chromadb
    from pathlib import Path
    
    # Bước 1: Kết nối tới ChromaDB local vector store
    db_path = Path(__file__).parent.parent / "data" / "chroma_db"
    
    # Nếu DB chưa được tạo (Task 4 chưa chạy), trả về list rỗng ngay lập tức
    # để tránh tải model nặng làm treo test.
    if not db_path.exists():
        print("Vector database chưa tồn tại. Vui lòng chạy Task 4 trước!")
        return []
        
    # Bước 2: Embed query bằng cùng model ở Task 4
    from sentence_transformers import SentenceTransformer
    # Lưu ý: Model này rất nhẹ nên sẽ được tải cực kỳ nhanh
    print("Đang tải model all-MiniLM-L6-v2 để embed câu query...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    query_embedding = model.encode(query).tolist()
    
        
    client = chromadb.PersistentClient(path=str(db_path))
    
    try:
        collection = client.get_collection(name="law_and_news_docs")
    except Exception:
        print("Collection 'law_and_news_docs' chưa tồn tại. Vui lòng chạy lại Task 4!")
        return []
    
    # Bước 3: Truy vấn vector store để lấy top_k kết quả
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )
    
    output = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            output.append({
                "content": doc,
                # L2 distance mặc định của ChromaDB: khoảng cách càng nhỏ càng giống nhau
                # Ta biến đổi distance thành similarity score (từ 0 -> 1)
                "score": 1.0 / (1.0 + float(dist)), 
                "metadata": meta
            })
            
    # Sắp xếp kết quả theo điểm số giảm dần
    output = sorted(output, key=lambda x: x["score"], reverse=True)
    return output


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
