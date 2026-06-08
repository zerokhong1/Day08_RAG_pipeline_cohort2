import urllib.request
from markitdown import MarkItDown

def test_download():
    url = "https://thcshbt-xuanhoa.hcm.edu.vn/uploads/news/2024_03/73-2021-qh14-luat-phong-chong-ma-tuy253202414.pdf"
    filename = "luat_phong_chong_ma_tuy_2021_hcm.pdf"
    
    md = MarkItDown()
    
    print(f"Downloading {filename} from {url}...")
    try:
        req = urllib.request.Request(
            url, 
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            with open(filename, 'wb') as f:
                f.write(response.read())
        print(f"Downloaded. Now testing text extraction...")
        
        result = md.convert(filename)
        text = result.text_content
        print(f"Extracted length: {len(text)}")
        print("First 500 chars:")
        print(text[:500])
        print("---------------------------------------\n")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_download()
