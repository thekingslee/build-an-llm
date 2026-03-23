import pymupdf
import pytesseract
from PIL import Image
import pdfplumber
import os
import sys
from pathlib import Path

 
def pdf_extract(file):
    with pdfplumber.open(file) as files:
        string = ''
        for page in files.pages: #for each page in the read pdf file
            pages_text = page.extract_text() or '' #extract texts from each page
            
            for line in pages_text.splitlines(): #splittng the lines in each page with splitlines()
                string += ' '.join(line.split()) + ' ' # .split() further split each line into words and a space, then joining it to the string
            string += '\n' #adding space before new page
    return string.strip()


 
def extract_text_with_ocr(file):
    string = ""
    doc = pymupdf.open(file)
    for page in doc:
        pix = page.get_pixmap()  # Render page to an image
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        ocr_result = pytesseract.image_to_string(img)  # Perform OCR
        # Post-process to add spaces where necessary
        string += ' '.join(ocr_result.split()) + ' '  # This removes extra spaces and adds single spaces
    return string


 
def save_text_to_file(string, filename):
    # Create the directory if it doesn't exist
    os.makedirs('scraped_pages_pdf', exist_ok=True)
    
    with open(f'scraped_pages_pdf/{filename}', 'w', encoding='utf-8') as f:
        f.write(string)

 
def main(file):  
    # First try to extract text using pdfplumber
    text = pdf_extract(file)
    
    # If no text is found, use OCR
    if not text.strip():
        print("No text found, performing OCR...")
        text = extract_text_with_ocr(file)

#    print("Extracted Text:")
#    print(text)

    # Extract the filename from the URL
    filename = os.path.basename(file).replace('.pdf', '.txt')  # Change extension to .txt

    save_text_to_file(text, filename)


 
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 scrape_local_pdf.py <path_to_pdf_file, including '/' at the end if it's a directory>")
        sys.exit(1)
    # Specify the directory you want to iterate over
    file_path = sys.argv[1] 
    directory_path = Path(file_path)  # Get the parent directory of the file path
    for file in directory_path.iterdir():
        if file.is_file() and file.suffix == '.pdf':  # Check if it's a PDF file (not a directory)
            main(file) # Get the file path from command line arguments