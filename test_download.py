import urllib.request
import os
from markitdown import MarkItDown

def test_download():
    urls = {
        "luat_phong_chong_ma_tuy_2021.pdf": "https://sonla.gov.vn/files/2-luat-phong-chong-ma-tuy-2021-luat-so-73-2021-qh14.pdf",
        "nghi_dinh_105_2021.pdf": "http://csnd.khanhhoa.gov.vn/uploads/tai-lieu/nghi-dinh-105-2021-nd-cp.pdf"
    }
    
    md = MarkItDown()
    
    for filename, url in urls.items():
        print(f"Downloading {filename} from {url}...")
        try:
            # Download with a custom user agent to avoid bot protection
            req = urllib.request.Request(
                url, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            )
            with urllib.request.urlopen(req, timeout=15) as response:
                with open(filename, 'wb') as f:
                    f.write(response.read())
            print(f"Downloaded {filename}. Now testing text extraction...")
            
            result = md.convert(filename)
            text = result.text_content
            print(f"Extracted length: {len(text)}")
            print("First 200 chars:")
            print(text[:200])
            print("---------------------------------------\n")
            
        except Exception as e:
            print(f"Error for {filename}: {e}")

if __name__ == "__main__":
    test_download()
