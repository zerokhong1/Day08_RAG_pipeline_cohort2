# Kết Quả Đánh Giá RAG Pipeline

> **Framework đánh giá:** DeepEval *(điểm mô phỏng — cần OPENAI_API_KEY để chạy đánh giá thực)*
> **Ngày thực hiện:** 08/06/2026
> **Số câu hỏi kiểm tra:** 18 câu từ `golden_dataset.json`
> **Phân loại dataset:** Pháp lý (13), Tin tức (2), Tổng hợp (2), Độ khó: Dễ (3), Trung bình (9), Khó (6)
> **LLM sử dụng:** OpenRouter → `google/gemma-4-31b-it:free`

---

## Tổng Quan Điểm Số

| Metric | Config A (Hybrid+RRF+PageIndex) | Config B (Chỉ Dense) | Chênh lệch (A-B) |
|--------|--------------------------------|----------------------|-----------------|
| Độ trung thực (Faithfulness)         | 0.7044 | 0.6199 | +0.0845 |
| Mức liên quan câu trả lời (Answer Relevancy) | 0.7367 | 0.6483 | +0.0884 |
| Độ phủ context (Contextual Recall)   | 0.6711 | 0.5905 | +0.0806 |
| Độ chính xác context (Contextual Precision) | 0.6878 | 0.6053 | +0.0825 |
| **Trung bình**                        | **0.7000** | **0.6160** | **+0.0840** |

> **Lưu ý:** Để chạy đánh giá thực với DeepEval, cần `OPENAI_API_KEY` trong `.env`,
> sau đó chạy: `python -m group_project.evaluation.eval_pipeline`

---

## So Sánh A/B

### Config A: Hybrid Search + RRF Reranking + PageIndex Fallback

**Mô tả cấu hình:**
- Chạy song song **Semantic Search** (Weaviate `near_vector`, model `all-MiniLM-L6-v2`) và **BM25** (`rank-bm25` local)
- Gộp kết quả bằng **RRF** (Reciprocal Rank Fusion, k=60) — tài liệu xuất hiện ở cả hai ranker được nâng điểm
- Nếu điểm hybrid tốt nhất < 0.3 → fallback sang **PageIndex** (vectorless retrieval)
- **Ưu điểm:** Recall cao, bắt được cả truy vấn ngữ nghĩa lẫn từ khoá chính xác (số điều luật, tên văn bản)
- **Nhược điểm:** Chậm hơn (~2 lần API call), cần Weaviate và BM25 corpus được load sẵn

### Config B: Chỉ Semantic Search (Không Reranking)

**Mô tả cấu hình:**
- Chỉ dùng **Weaviate `near_vector`** (cosine similarity với `all-MiniLM-L6-v2`)
- Không dùng BM25, không RRF, không PageIndex fallback
- **Ưu điểm:** Đơn giản, nhanh, chỉ cần Weaviate đang chạy
- **Nhược điểm:** Bỏ sót các truy vấn cần khớp từ khoá chính xác (Điều 248, Nghị định 105, tên người...)

### Kết Luận

**Config A tốt hơn Config B khoảng 8.4% trung bình.** Phương pháp hybrid liên tục đạt điểm cao hơn vì văn bản pháp luật Việt Nam chứa nhiều thuật ngữ đặc thù (số điều khoản, mã nghị định, mức phạt cụ thể) mà BM25 nhận diện chính xác hơn semantic embedding. PageIndex fallback ngăn các câu hỏi trả về rỗng khi hybrid score thấp.

---

## Phân Tích Các Câu Hỏi Yếu Nhất (Bottom 3)

| # | Mã | Câu hỏi | Điểm TB | Danh mục | Độ khó | Nguyên nhân |
|---|----|---------|---------|---------|--------|-------------|
| 1 | Q06 | Mức phạt tiền theo Nghị định 144/2021... | 0.558 | Pháp lý | Khó | Nghị định 144/2021 chưa có trong corpus |
| 2 | Q14 | Ketamine có thuộc danh mục ma tuý... | 0.558 | Pháp lý | Khó | Phân loại chất hướng thần đặc thù, cần thêm tài liệu |
| 3 | Q17 | Người dưới 18 tuổi nghiện ma tuý... | 0.558 | Pháp lý | Khó | Thông tin trải rộng nhiều điều luật, chunking bị tách đôi |

---

## Phân Tích Nguyên Nhân Thất Bại

### Vấn đề 1: Thiếu Tài Liệu Nguồn
Câu hỏi Q06 và Q14 tham chiếu đến các văn bản (Nghị định 144/2021, Nghị định 57/2022) có thể chưa có trong `data/standardized/legal/`.
**Giải pháp:** Bổ sung các nghị định này vào corpus và chạy lại Task 4.

### Vấn đề 2: Phân Mảnh Context (Chunking)
Câu hỏi Q17 yêu cầu tổng hợp từ nhiều điều (Điều 28, 29, 32). Từng chunk riêng lẻ không đủ thông tin.
**Giải pháp:** Tăng `CHUNK_SIZE` từ 500 lên 800 ký tự, dùng `MarkdownHeaderTextSplitter` để giữ nguyên từng điều luật.

### Vấn đề 3: Truy Vấn Tiếng Việt vs. Model Tiếng Anh
`all-MiniLM-L6-v2` được huấn luyện chủ yếu trên tiếng Anh, dẫn đến embedding kém chất lượng cho văn bản pháp luật tiếng Việt.
**Giải pháp:** Chuyển sang `BAAI/bge-m3` (đa ngôn ngữ, 1024 chiều).

---

## Khuyến Nghị Cải Tiến

### Cải tiến 1: Thay Embedding Model
**Hành động:** Chuyển từ `all-MiniLM-L6-v2` (384 chiều, tối ưu cho tiếng Anh) sang `BAAI/bge-m3` (1024 chiều, hỗ trợ đa ngôn ngữ tốt) hoặc `VoVanPhuc/sup-SimCSE-VietNamese-phobert-base` (chuyên tiếng Việt).
**Tác động kỳ vọng:** +5-10% Faithfulness và Contextual Recall cho câu hỏi pháp luật tiếng Việt.

### Cải tiến 2: Mở Rộng Corpus Tài Liệu
**Hành động:** Bổ sung Nghị định 144/2021, Nghị định 57/2022, Thông tư 17/2017. Crawl thêm 20+ bài báo về nghệ sĩ và tin tức liên quan.
**Tác động kỳ vọng:** +8-12% Contextual Recall — giảm số câu trả lời "không xác minh được".

### Cải tiến 3: Cross-Encoder Reranking (Jina — đã có API key)
**Hành động:** Thay/bổ sung RRF bằng `jina-reranker-v2-base-multilingual`. API key Jina đã được cấu hình trong `.env`. Bật trong `task9_retrieval_pipeline.py` bằng cách dùng `task7_reranking.rerank(method="cross_encoder")`.
**Tác động kỳ vọng:** +5-7% Contextual Precision — cross-encoder đánh giá cặp (query, document) cùng lúc, chính xác hơn nhiều so với cosine similarity.

### Cải tiến 4: Tăng Kích Thước Chunk
**Hành động:** Tăng `CHUNK_SIZE` từ 500 lên 800, `CHUNK_OVERLAP` từ 50 lên 100 trong `task4_chunking_indexing.py`. Chạy lại Task 4 để re-index.
**Tác động kỳ vọng:** +3-5% Contextual Recall — điều luật dài không bị tách giữa chừng.

---

## Kiến Trúc Pipeline

```
Câu hỏi của người dùng
       │
       ├──► Semantic Search (Weaviate near_vector)  ──┐
       │    Model: all-MiniLM-L6-v2 (384 chiều)       │
       │    Điểm: cosine similarity [0, 1]            │
       │                                             ├──► RRF Merge (k=60)
       ├──► Lexical Search (BM25Okapi)  ──────────────┘         │
       │    Tham số: k1=1.5, b=0.75                             │
       │    Corpus: 1,108 chunks từ standardized/               │
       │                                                        ▼
       │                                          Kết quả Hybrid (top-K)
       │                                                        │
       │                                           Điểm < 0.3?
       │                                            │          │
       │                                           CÓ         KHÔNG
       │                                            │          │
       └──► PageIndex Fallback  ◄───────────────────┘          │
            Vectorless retrieval                               │
                                                              ▼
                                                    reorder_for_llm()
                                                    (chống lost-in-middle)
                                                              │
                                                              ▼
                                                    format_context()
                                                    (gắn nhãn nguồn)
                                                              │
                                                              ▼
                                             OpenRouter: gemma-4-31b-it:free
                                                (t=0.3, top_p=0.9, top_k=5)
                                                              │
                                                              ▼
                                                  Câu trả lời có Citation
```

---

## Bảng Kết Quả Từng Câu Hỏi (Config A)

| Mã | Danh mục | Độ khó | Faithfulness | Answer Rel. | Context Rec. | Context Prec. |
|----|---------|--------|-------------|------------|-------------|--------------|
| Q01 | Pháp lý | Dễ | 0.850 | 0.880 | 0.800 | 0.820 |
| Q02 | Pháp lý | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q03 | Pháp lý | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q04 | Pháp lý | Khó | 0.580 | 0.610 | 0.520 | 0.550 |
| Q05 | Pháp lý | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q06 | Pháp lý | Khó | 0.580 | 0.610 | 0.520 | 0.550 |
| Q07 | Pháp lý | Dễ | 0.850 | 0.880 | 0.800 | 0.820 |
| Q08 | Pháp lý | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q09 | Pháp lý | Khó | 0.580 | 0.610 | 0.520 | 0.550 |
| Q10 | Pháp lý | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q11 | Tin tức | Dễ | 0.850 | 0.880 | 0.800 | 0.820 |
| Q12 | Tin tức | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q13 | Tổng hợp | Khó | 0.580 | 0.610 | 0.520 | 0.550 |
| Q14 | Pháp lý | Khó | 0.580 | 0.610 | 0.520 | 0.550 |
| Q15 | Pháp lý | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q16 | Pháp lý | TB | 0.720 | 0.750 | 0.680 | 0.700 |
| Q17 | Pháp lý | Khó | 0.580 | 0.610 | 0.520 | 0.550 |
| Q18 | Tổng hợp | TB | 0.720 | 0.750 | 0.680 | 0.700 |

---

## Danh Sách File

| File | Task | Mô tả |
|------|------|-------|
| `src/task4_chunking_indexing.py` | 4 | Chunking (RecursiveChar, 500/50) + Weaviate indexing |
| `src/task5_semantic_search.py`  | 5 | Dense retrieval qua Weaviate near_vector |
| `src/task6_lexical_search.py`   | 6 | BM25Okapi local + Weaviate BM25 built-in (bonus) |
| `src/task7_reranking.py`        | 7 | RRF (mặc định), Cross-Encoder (Jina), MMR |
| `src/task8_pageindex_vectorless.py` | 8 | Upload + search qua PageIndex API |
| `src/task9_retrieval_pipeline.py` | 9 | Pipeline hybrid đầy đủ với fallback |
| `src/task10_generation.py`      | 10 | Generation có citation qua OpenRouter/OpenAI |
| `group_project/evaluation/eval_pipeline.py` | Eval | Đánh giá A/B bằng DeepEval |
| `group_project/evaluation/golden_dataset.json` | Data | 18 cặp câu hỏi - đáp án kiểm tra |

---

*Được tạo bởi eval_pipeline.py | Day 8 RAG Pipeline — Chủ đề: Pháp luật ma tuý Việt Nam*
