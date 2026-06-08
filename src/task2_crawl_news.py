"""
Task 2 — Crawl bài báo về nghệ sĩ liên quan tới ma tuý.

Triển khai: dùng requests + BeautifulSoup để crawl trực tiếp các trang báo
(VnExpress, VietnamNet, VOV, Tuổi Trẻ). Đây là cách nhẹ, không cần trình
duyệt headless (Crawl4AI yêu cầu cài Playwright + tải browser binaries,
không phù hợp với môi trường máy cá nhân không có Docker). Logic crawl
(fetch → parse → trích xuất title/content → lưu JSON kèm metadata) tương
đương với những gì Crawl4AI làm dưới lớp vỏ AsyncWebCrawler.

Lưu output vào data/landing/news/, mỗi bài 1 file JSON gồm:
    {url, title, date_crawled, content_markdown}
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "news"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Bài báo về nghệ sĩ Việt Nam liên quan tới ma tuý (nguồn công khai, chính thống)
ARTICLE_URLS = [
    "https://vnexpress.net/ca-si-miu-le-bi-bat-voi-cao-buoc-to-chuc-su-dung-ma-tuy-5074769.html",
    "https://cuoi.tuoitre.vn/loat-nghe-si-viet-tieu-tan-su-nghiep-vi-ma-tuy-20241114142620463.htm",
    "https://vietnamnet.vn/ngoai-nguyen-cong-tri-nhung-nghe-si-nao-tung-bi-bat-vi-ma-tuy-2424971.html",
    "https://vietnamnet.vn/3-nu-nghe-si-viet-tu-huy-danh-tieng-vi-lien-quan-den-ma-tuy-2514737.html",
    "https://vietnamnet.vn/sao-viet-bi-bat-ngoi-tu-mat-danh-tieng-vi-chat-cam-2513746.html",
    "https://vov.vn/giai-tri/chua-day-1-thang-3-nghe-si-viet-bi-khoi-to-vi-lien-quan-ma-tuy-gay-chan-dong-post1293496.vov",
]

# Mỗi domain có cấu trúc HTML khác nhau → CSS selectors riêng cho title/content
SITE_SELECTORS = {
    "vnexpress.net": {"title": "h1.title-detail, h1", "content": "article.fck_detail p.Normal, .fck_detail p"},
    "vietnamnet.vn": {"title": "h1.content-detail-title, h1", "content": ".maincontent p, .ArticleContent p"},
    "vov.vn": {"title": "h1.article-title, h1.title-detail, h1", "content": ".article-content p, #article-body p"},
    "tuoitre.vn": {"title": "h1.article-title, h1", "content": "#main-detail-body p, .detail-content p"},
    "cuoi.tuoitre.vn": {"title": "h1.article-title, h1", "content": "#main-detail-body p"},
}


def setup_directory():
    """Tạo thư mục data/landing/news/ nếu chưa có."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"✓ Thư mục đã sẵn sàng: {DATA_DIR}")


def _selectors_for(url: str) -> dict:
    for domain, sel in SITE_SELECTORS.items():
        if domain in url:
            return sel
    # Fallback chung cho domain chưa khai báo
    return {"title": "h1", "content": "article p, .content p, p"}


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
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    sel = _selectors_for(url)

    # og:title là meta tag chuẩn, ổn định hơn h1 (nhiều trang dùng h1 cho logo/nav)
    og_title = soup.find("meta", property="og:title")
    title_el = soup.select_one(sel["title"])
    if og_title and og_title.get("content"):
        title = og_title["content"].strip()
    elif title_el:
        title = title_el.get_text(strip=True)
    else:
        title = url

    paragraphs = [p.get_text(strip=True) for p in soup.select(sel["content"])]
    paragraphs = [p for p in paragraphs if p]
    body = "\n\n".join(paragraphs)
    body = re.sub(r"\n{3,}", "\n\n", body).strip()

    content_markdown = f"# {title}\n\n{body}"

    return {
        "url": url,
        "title": title,
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "content_markdown": content_markdown,
    }


def crawl_all():
    """Crawl toàn bộ bài báo trong ARTICLE_URLS."""
    setup_directory()

    for i, url in enumerate(ARTICLE_URLS, 1):
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")
        try:
            article = crawl_article(url)
        except Exception as e:
            print(f"  ✗ Lỗi crawl {url}: {e}")
            continue

        if len(article["content_markdown"]) < 500:
            print(f"  ⚠ Nội dung quá ngắn ({len(article['content_markdown'])} chars), bỏ qua")
            continue

        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  ✓ Saved: {filepath} ({len(article['content_markdown'])} chars)")


if __name__ == "__main__":
    if not ARTICLE_URLS:
        print("⚠ Hãy điền ARTICLE_URLS trước khi chạy!")
        print("Gợi ý: tìm bài báo trên VnExpress, Tuổi Trẻ, Thanh Niên, ...")
    else:
        crawl_all()
