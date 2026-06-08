import os
import sys
from pathlib import Path

# === THÊM ĐOẠN NÀY VÀO ĐỂ ẨN LOG WARNING PHIỀN PHỨC ===
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# =====================================================

import streamlit as st
from dotenv import load_dotenv
import requests

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

# Import trực tiếp hàm RAG generation từ Task 10
from src.task10_generation import generate_with_citation
from src.task4_chunking_indexing import CHUNKING_METHOD, CHUNK_SIZE, CHUNK_OVERLAP

load_dotenv(ROOT_DIR / ".env")


def build_follow_up_prompt(query: str, history: list[dict]) -> str:
    if not history:
        return query

    history_lines = []
    for i, turn in enumerate(history[-5:], start=1):
        history_lines.append(
            f"Turn {i} - User: {turn['question']}\nAssistant: {turn['answer']}"
        )

    history_text = "\n\n".join(history_lines)
    return (
        f"Conversation history:\n{history_text}\n\n"
        f"Current question: {query}"
    )


def init_session_state() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def render_source_card(source: dict, index: int) -> None:
    """Hiển thị nguồn tài liệu dưới dạng Card giao diện đẹp mắt và an toàn với mọi Theme"""
    metadata = source.get("metadata", {})
    source_name = metadata.get("source", "Tài liệu chưa rõ nguồn")
    source_type = metadata.get("type", "Chưa phân loại")
    score = source.get("score", 0.0)
    excerpt = source.get("content", "").strip()

    # Dùng container viền ngoài của Streamlit, tự động thay đổi màu nền và màu chữ theo Theme (Sáng/Tối)
    with st.container(border=True):
        st.markdown(f"##### 📌 Nguồn {index}: {source_name}")
        st.caption(f"Loại: {source_type}  •  Độ liên quan (Score): :red[{score:.3f}]")
        st.markdown(f"*{excerpt}*")


def main() -> None:
    # Cấu hình trang với giao diện rộng rãi hơn
    st.set_page_config(page_title="RAG Chatbot Luật Ma túy", page_icon="⚖️", layout="wide")
    
    init_session_state()

    # Kiểm tra trạng thái cấu hình OpenAI API Key để hiển thị lên Sidebar
    openai_key = os.getenv("OPENAI_API_KEY", "").strip()
    is_local_mode = not openai_key

    # --- CỘT TRÁI: THÔNG TIN TRỢ LÝ ---
    sidebar, chat_zone = st.columns([1, 2.5], gap="large")

    with sidebar:
        st.logo("https://cdn-icons-png.flaticon.com/512/3135/3135671.png", icon_image="https://lapphap.vn/Upload/PH%C3%81P-L%C3%9D.jpg")
        st.title("⚖️ Trợ Lý Pháp Luật")
        
        if is_local_mode:
            st.caption("Chế độ: **Mô hình Local Qwen2.5:7b (Ollama)**")
            st.warning("Không tìm thấy `OPENAI_API_KEY`. Đã tự động chuyển sang chạy mô hình Local để tiết kiệm chi phí và bảo mật.")
        else:
            st.caption("🌐 Chế độ: **OpenAI GPT Cloud**")
            st.success("✨ Đã nhận diện `OPENAI_API_KEY`. Hệ thống đang sử dụng LLM Cloud.")

        st.info(
            "💡 **Phạm vi hỗ trợ:**\n"
            "Chuyên giải đáp về Pháp luật phòng chống Ma túy, hình phạt tội phạm ma túy và các tin tức xã hội liên quan."
        )
        
        with st.expander("🛠️ Chi tiết hệ thống", expanded=False):
            st.markdown(
                f"- **Giao diện:** Streamlit Chat UI  \n"
                f"- **Mô hình hiện tại:** {'Qwen2.5:7b (Ollama)' if is_local_mode else 'OpenAI GPT'}  \n"
                f"- **Phân mảnh (Strategy):** `{CHUNKING_METHOD}`  \n"
                f"- **Kích thước (Chunk Size):** `{CHUNK_SIZE}` ký tự  \n"
                f"- **Độ chồng lấp (Overlap):** `{CHUNK_OVERLAP}` ký tự  \n"
                f"- **Kiểm soát phạm vi:** Tự động từ chối câu hỏi không liên quan."
            )
            
        # Nút xóa lịch sử chat nằm gọn gàng bên cột trái
        if st.button("🔄 Xóa lịch sử hội thoại", use_container_width=True, type="secondary"):
            st.session_state.history = []
            st.rerun()

    # --- CỘT PHẢI: KHUNG CHÁT CHÍNH ---
    with chat_zone:
        st.subheader("💬 Hộp thoại Tư vấn Pháp luật")
        
        # 1. Hiển thị Lịch sử chat theo dạng dòng thời gian hiện đại (st.chat_message)
        for turn in st.session_state.history:
            with st.chat_message("user"):
                st.markdown(turn["question"])
                
            with st.chat_message("assistant", avatar="⚖️"):
                st.markdown(turn["answer"])
                if turn.get("retrieval_source") and turn["retrieval_source"] != "error":
                    with st.expander(f"📚 Xem đầy đủ {len(turn.get('sources', []))} nguồn tài liệu trích dẫn", expanded=False):
                        st.caption(f"Phương thức tìm kiếm: {turn['retrieval_source']}")
                        for i, source in enumerate(turn.get("sources", []), start=1):
                            render_source_card(source, i)

        # 2. Ô nhập nội dung chat cố định ở đáy màn hình
        query = st.chat_input("Nhập câu hỏi của bạn về luật phòng chống ma túy tại đây...")

        if query:
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant", avatar="⚖️"):
                spinner_msg = "Đang xử lý bằng Qwen2.5:7b Local..." if is_local_mode else "Đang xử lý bằng OpenAI..."
                with st.spinner(spinner_msg):
                    prompt = build_follow_up_prompt(query.strip(), st.session_state.history)
                    try:
                        if is_local_mode:
                            # THỦ TỤC RUN LOCAL: Lấy nguồn tài liệu trích dẫn trước (Retrieval)
                            result = generate_with_citation(prompt, top_k=5)
                            sources = result.get("sources", [])
                            retrieval_source = result.get("retrieval_source", "hybrid")
                            
                            # Xây dựng ngữ cảnh từ tài liệu tìm được để chuyển sang Ollama
                            context_str = ""
                            if sources:
                                context_str = "\n".join([f"- Tài liệu {i+1}: {s.get('content', '')}" for i, s in enumerate(sources)])
                            
                            # Cấu hình luật kiểm soát câu hỏi liên quan nghiêm ngặt cho mô hình local
                            full_prompt_with_rag = (
                                f"Bạn là một trợ lý pháp luật chuyên nghiệp, chỉ có nhiệm vụ giải đáp về chủ đề 'Pháp luật phòng chống Ma túy, hình phạt tội phạm ma túy và các tin tức xã hội liên quan'.\n"
                                f"QUY TẮC BẮT BUỘC:\n"
                                f"1. Nếu câu hỏi từ người dùng KHÔNG liên quan đến chủ đề pháp luật ma túy hoặc tin tức liên quan (ví dụ: hỏi về lập trình, nấu ăn, thời tiết, toán học, hoặc các chủ đề pháp luật khác không dính dáng tới ma túy), bạn phải ngay lập tức TỪ CHỐI TRẢ LỜI và xin lỗi người dùng một cách lịch sự (Ví dụ: 'Xin lỗi, tôi là trợ lý chuyên về pháp luật ma túy nên không thể trả lời câu hỏi này...').\n"
                                f"2. Nếu câu hỏi liên quan, hãy dựa vào ngữ cảnh tài liệu dưới đây để tổng hợp câu trả lời chính xác bằng tiếng Việt.\n\n"
                                f"Ngữ cảnh tài liệu:\n{context_str}\n\n"
                                f"Yêu cầu xử lý từ hội thoại:\n{prompt}\n\n"
                                f"Câu trả lời:"
                            )

                            # Gọi request trực tiếp đến endpoint của Ollama Local với model qwen2.5:7b
                            ollama_url = "http://localhost:11434/api/generate"
                            payload = {
                                "model": "qwen2.5:7b",
                                "prompt": full_prompt_with_rag,
                                "stream": False
                            }
                            
                            ollama_response = requests.post(ollama_url, json=payload, timeout=90)
                            ollama_response.raise_for_status()
                            answer = ollama_response.json().get("response", "Không nhận được phản hồi từ Ollama.")
                        else:
                            # THỦ TỤC RUN CLOUD: Gọi trực tiếp toàn bộ pipeline Task 10 bằng OpenAI
                            result = generate_with_citation(prompt, top_k=5)
                            answer = result.get("answer", "Không nhận được phản hồi.")
                            sources = result.get("sources", [])
                            retrieval_source = result.get("retrieval_source", "hybrid")
                            
                    except Exception as exc:
                        answer = (
                            "⚠ Không thể hoàn thành xử lý câu trả lời.\n\n"
                            f"*Chi tiết lỗi:* {exc}"
                        )
                        sources = []
                        retrieval_source = "error"

                    # Hiển thị câu trả lời mới sinh lên màn hình
                    st.markdown(answer)
                    if retrieval_source != "error":
                        with st.expander(f"📚 Xem đầy đủ {len(sources)} nguồn tài liệu trích dẫn", expanded=False):
                            st.caption(f"Phương thức tìm kiếm: {retrieval_source}")
                            for i, source in enumerate(sources, start=1):
                                render_source_card(source, i)

            # Lưu lượt hội thoại mới vào bộ nhớ session_state để duy trì bộ nhớ ngữ cảnh (context)
            st.session_state.history.append(
                {
                    "question": query.strip(),
                    "answer": answer,
                    "sources": sources,
                    "retrieval_source": retrieval_source,
                }
            )
            
            # Khởi động lại luồng để Streamlit cập nhật đồng bộ giao diện
            st.rerun()


if __name__ == "__main__":
    main()