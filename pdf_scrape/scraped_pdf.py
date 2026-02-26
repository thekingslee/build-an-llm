import requests
import pymupdf
import pytesseract
from PIL import Image
import pdfplumber
import io
import os
from urllib.parse import urlparse
import sys  # Import sys to handle command-line arguments

def download_pdf(url):
    response = requests.get(url)
    response.raise_for_status()  # Check for request errors
    return response.content

def extract_text_from_pdf(pdf_content):
    text = ""
    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            # Split the page text into lines and then join words with a single space
            for line in page_text.splitlines():
                text += ' '.join(line.split()) + ' '  # Join words in the line with a single space
            text += '\n'  # Optionally add a newline after each page
    return text.strip()  # Remove trailing whitespace

def extract_text_with_ocr(pdf_content):
    text = ""
    doc = pymupdf.open(io.BytesIO(pdf_content))
    for page in doc:
        pix = page.get_pixmap()  # Render page to an image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result = pytesseract.image_to_string(img)  # Perform OCR
        # Post-process to add spaces where necessary
        text += ' '.join(ocr_result.split())  # This removes extra spaces and adds single spaces
    return text

def save_text_to_file(text, filename):
    # Create the directory if it doesn't exist
    os.makedirs('scraped_pages_pdf', exist_ok=True)
    
    with open(f'scraped_pages_pdf/{filename}', 'w', encoding='utf-8') as f:
        f.write(text)

def main(url):
    pdf_content = download_pdf(url)
    
    # First try to extract text using pdfplumber
    text = extract_text_from_pdf(pdf_content)
    
    # If no text is found, use OCR
    if not text.strip():
        print("No text found, performing OCR...")
        text = extract_text_with_ocr(pdf_content)

    print("Extracted Text:")
    print(text)

    # Extract the filename from the URL
    parsed_url = urlparse(url)
    filename = os.path.basename(parsed_url.path).replace('.pdf', '.txt')  # Change extension to .txt

    save_text_to_file(text, filename)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test.py <PDF_URL>")
        sys.exit(1)

    pdf_url = sys.argv[1]  # Get the URL from command line arguments
    main(pdf_url)