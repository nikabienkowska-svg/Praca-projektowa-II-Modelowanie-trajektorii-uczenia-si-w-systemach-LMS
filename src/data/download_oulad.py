import urllib.request
import zipfile
import os
import ssl

def download_and_extract():
    # Pominięcie weryfikacji SSL (w razie problemów z certyfikatami)
    ssl._create_default_https_context = ssl._create_unverified_context
    
    url = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
    zip_path = "data/oulad.zip"
    extract_to = "data/raw/"
    
    os.makedirs(extract_to, exist_ok=True)
    
    print(f"Pobieranie danych z {url}...")
    print("To może chwilę potrwać (ok. 45 MB). Proszę czekać...")
    
    urllib.request.urlretrieve(url, zip_path)
    print("Pobrano. Rozpakowywanie...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        
    os.remove(zip_path)
    print(f"Gotowe! Pliki CSV zostały rozpakowane do folderu '{extract_to}'.")

if __name__ == "__main__":
    download_and_extract()
