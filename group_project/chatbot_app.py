import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT_DIR))

from src.task10_generation import generate_with_citation

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
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""


def render_source(source: dict, index: int) -> None:
    metadata = source.get("metadata", {})
    source_name = metadata.get("source", "unknown")
    source_type = metadata.get("type", "unknown")
    score = source.get("score", 0.0)
    excerpt = source.get("content", "").strip().replace("\n", " ")[:280]

    st.markdown(
        f"**{index}. {source_name}**  \n"
        f"_Type: {source_type} · Score: {score:.3f}_  \n"
        f"{excerpt}..."
    )


def main() -> None:
    st.set_page_config(page_title="RAG Chatbot Luật Ma túy", page_icon="⚖️")
    st.title("RAG Chatbot — Pháp luật Ma túy & Tin tức liên quan")

    st.markdown(
        """
        Chatbot này dùng pipeline retrieval + generation để trả lời câu hỏi
        bằng tiếng Việt với trích dẫn nguồn. Bạn có thể hỏi luật pháp,
        hình phạt, thông tin báo chí, và tiếp tục với các câu hỏi follow-up.
        """
    )

    st.markdown(
        "- Sử dụng `Streamlit` cho giao diện chat.  \n"
        "- Kết quả có citation.  \n"
        "- Lưu lại bộ nhớ hội thoại trong phiên làm việc.  \n"
        "- Hiển thị source documents đã dùng."
    )

    init_session_state()

    with st.expander("Yêu cầu môi trường"):
        st.write("- Tạo file `.env` ở thư mục gốc với `OPENAI_API_KEY`.")
        st.write("- Chạy: `streamlit run group_project/chatbot_app.py`")

    query = st.text_area("Nhập câu hỏi của bạn", value=st.session_state.chat_input, height=140)
    submit = st.button("Gửi câu hỏi")

    if submit and query.strip():
        st.session_state.chat_input = ""
        with st.spinner("Đang truy vấn và tổng hợp câu trả lời... 🤖"):
            prompt = build_follow_up_prompt(query.strip(), st.session_state.history)
            try:
                result = generate_with_citation(prompt, top_k=5)
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                retrieval_source = result.get("retrieval_source", "hybrid")
            except Exception as exc:
                answer = (
                    "⚠ Không thể gọi LLM: kiểm tra OPENAI_API_KEY hoặc kết nối mạng.\n"
                    f"Lỗi: {exc}"
                )
                sources = []
                retrieval_source = "error"

        st.session_state.history.append(
            {
                "question": query.strip(),
                "answer": answer,
                "sources": sources,
                "retrieval_source": retrieval_source,
            }
        )

    if st.session_state.history:
        st.markdown("---")
        st.header("Lịch sử hội thoại")
        for idx, turn in enumerate(reversed(st.session_state.history), start=1):
            st.markdown(f"**User:** {turn['question']}")
            st.info(turn['answer'])
            if turn.get("retrieval_source"):
                st.caption(f"Retrieval source: {turn['retrieval_source']}")

    if submit and st.session_state.history:
        latest = st.session_state.history[-1]
        if latest.get("retrieval_source") != "error":
            st.markdown("---")
            st.subheader("Sources được dùng")
            for i, source in enumerate(latest.get("sources", []), start=1):
                render_source(source, i)

    if st.button("Xóa bộ nhớ hội thoại"):
        st.session_state.history = []
        st.rerun()


if __name__ == "__main__":
    main()
