"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Hướng dẫn:
    1. Crawl tối thiểu 5 bài báo từ các trang tin tức Việt Nam.
    2. Sử dụng Crawl4AI hoặc thư viện crawling tương tự.
    3. Lưu output vào data/landing/news/
    4. Mỗi bài lưu 1 file JSON với metadata (url, title, date_crawled, content).

Cài đặt:
    pip install crawl4ai
"""

import json
import requests
from datetime import datetime
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


# TODO: Điền danh sách URL bài báo cần crawl
ARTICLE_URLS = [
    "https://tienphong.vn/tu-ket-qua-xet-nghiem-5-loai-ma-tuy-cua-ngoc-son-va-dong-thai-cua-nhieu-nghe-si-post1845555.tpo",
    "https://znews.vn/hau-qua-nghiem-trong-khi-nghe-si-viet-lien-tuc-vuong-on-ao-ma-tuy-post1650870.html",
    "https://tienphong.vn/nhung-ca-si-dien-vien-bi-ma-tuy-tan-pha-post1528839.tpo",
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://vietnamnet.vn/nam-ca-si-long-nhat-vua-bi-khoi-to-bat-tam-giam-vi-ma-tuy-2517561.html",
    "https://kenh14.vn/sao-viet-tieu-tan-su-nghiep-vi-lien-quan-den-ma-tuy-215260522111209355.chn",
    "https://nld.com.vn/ca-si-miu-le-bi-khoi-to-tam-giam-ve-toi-to-chuc-su-dung-trai-phep-chat-ma-tuy-196260516215034895.htm",
    "https://www.24h.com.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-c731a1763370.htm",
]


def crawl_article(url: str) -> dict:
    """
    Crawl một bài báo và trả về dict chứa metadata + content.

    Returns:
        {
            "url": str,
            "title": str,
            "date_crawled": str (ISO format),
            "content_markdown": str
        }
    """
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    
    return {
        "url": url,
        "title": "News Article",
        "date_crawled": datetime.now().isoformat(),
        "content_markdown": response.text,  # Tạm lưu HTML, Task 3 sẽ convert sau
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = crawl_article(url)
            filename = f"article_{i:02d}.json"
            filepath = DATA_DIR / filename
            filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  ✓ Saved: {filepath}")
        except Exception as e:
            print(f"  ✗ Lỗi khi crawl {url}: {e}")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        crawl_all()
