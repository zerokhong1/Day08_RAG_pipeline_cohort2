"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Đã tải về 4 văn bản pháp luật chính thống (file gốc PDF) — lưu vào
data/landing/legal/. Nguồn: datafiles.chinhphu.vn (Cổng TTĐT Chính phủ) và
g7.cdnchinhphu.vn (Công báo Chính phủ — congbao.chinhphu.vn):

    1. Luật Phòng, chống ma tuý 2021 (Luật số 73/2021/QH14)
       https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf
    2. Văn bản hợp nhất Luật Phòng, chống ma tuý (số 117/VBHN-VPQH, 2025)
       — hợp nhất Luật PCMT 2021 cùng các sửa đổi, thay cho việc tải Nghị
       định 105/2021 (file "*.signed*.pdf" của CP chỉ là bản scan ảnh,
       MarkItDown không trích xuất được text — xem ghi chú bên dưới).
       https://g7.cdnchinhphu.vn/.../2025_1291+1292_117-VBHN-VPQH.pdf
    3. Bộ luật Hình sự 2015 (văn bản hợp nhất, sửa đổi 2017) — Chương XX
       https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf
    4. Văn bản hợp nhất Luật Xử lý vi phạm hành chính (số 63/VBHN-VPQH, 2025)
       — quy định khung xử phạt hành chính áp dụng cho các vi phạm liên
       quan ma tuý chưa đến mức truy cứu hình sự; thay cho Nghị định
       57/2022 (cũng là file scan ảnh, không trích xuất được text).
       https://g7.cdnchinhphu.vn/.../2025_1097+1098_63-VBHN-VPQH.pdf

GHI CHÚ: Các file PDF "*.signed*.pdf" trên datafiles.chinhphu.vn (vd. Nghị
định 105/2021, Nghị định 57/2022) là bản SCAN ẢNH (đã kiểm tra bằng
pdfplumber: 0 ký tự trích xuất được, 1 ảnh / trang) — MarkItDown/pdfminer
không thể chuyển thành text mà không có OCR. Vì vậy đã thay bằng các "văn
bản hợp nhất" (VBHN) tương ứng từ Công báo Chính phủ — đây là các bản số
hoá có lớp text thật, đồng thời vẫn là văn bản pháp luật chính thống, hợp
lệ và liên quan trực tiếp tới chủ đề ma tuý.

Chạy lại để tải về (script idempotent — bỏ qua file đã tồn tại):
    python -m src.task1_collect_legal_docs
"""

from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

_VBHN_117_URL = (
    "https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbRV69z2fVk0hY5tYz5CN7quDEa_LFo2fCVspxWd3gtd-4e_IwGFMb6YvSF9zjz00X4-kkZImxN8Bg2hLuXFl7OKQ~~"
    "&file_name=2025_1291+%2b+1292_117-VBHN-VPQH.pdf"
)
_VBHN_63_URL = (
    "https://g7.cdnchinhphu.vn/api/download/stream?Url=tm-8mq6BhNw0NbrKRhTDAQWsKg3tuqaY0aWypnY78U6M2BY68Ekp0Gvvr483flbRGpGCikw-77mrTVjBEapvfJYMz29u5LY2hTNNKTmIomXNN2-v2PGGQWweEonwBR5daYKVznnr7L_iTq9OmKbSjQ~~"
    "&file_name=2025_1097+%2b+1098_63-VBHN-VPQH.pdf"
)

# Direct download links — file gốc PDF từ Cổng TTĐT Chính phủ / Công báo Chính phủ.
# (2) và (4) là các "văn bản hợp nhất" có lớp text thật — thay cho 2 file
# scan ảnh gốc (Nghị định 105/2021, Nghị định 57/2022) mà MarkItDown không
# trích xuất được nội dung.
DOCUMENTS = {
    "luat-phong-chong-ma-tuy-2021.pdf":
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2022/01/73luat.pdf",
    "vbhn-117-luat-phong-chong-ma-tuy.pdf": _VBHN_117_URL,
    "bo-luat-hinh-su-2015.pdf":
        "https://datafiles.chinhphu.vn/cpp/files/vbpq/2025/9/135-vbhn-vpqh.pdf",
    "vbhn-63-luat-xu-ly-vi-pham-hanh-chinh.pdf": _VBHN_63_URL,
}


def setup_directory():
    """Tạo thư mục data/landing/legal/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def download_file(url: str, filename: str):
    """Tải 1 file PDF từ direct link về DATA_DIR (bỏ qua nếu đã tồn tại)."""
    filepath = DATA_DIR / filename
    if filepath.exists() and filepath.stat().st_size > 1024:
        print(f"  ↷ Đã tồn tại, bỏ qua: {filepath.name}")
        return

    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        filepath.write_bytes(response.content)
    except requests.exceptions.SSLError:
        # Một số CDN của Chính phủ (vd. g7.cdnchinhphu.vn) có chuỗi chứng chỉ
        # mà bộ CA bundle mặc định của Python (certifi) chưa tin cậy, trong
        # khi schannel/curl trên Windows lại xác thực được bình thường — đã
        # kiểm chứng thủ công nội dung tải về là hợp lệ (PDF chính thống từ
        # Công báo Chính phủ). Dùng curl của hệ thống làm phương án dự phòng.
        import subprocess
        subprocess.run(
            ["curl", "-sL", "--ssl-no-revoke", "-A", HEADERS["User-Agent"],
             "-o", str(filepath), url],
            check=True, timeout=120,
        )

    print(f"  ✓ Đã tải: {filepath.name} ({filepath.stat().st_size:,} bytes)")


def collect_all():
    setup_directory()
    for filename, url in DOCUMENTS.items():
        print(f"Downloading {filename} <- {url}")
        try:
            download_file(url, filename)
        except Exception as e:
            print(f"  ✗ Lỗi tải {filename}: {e}")


if __name__ == "__main__":
    collect_all()
