import pymupdf
import pytesseract
from PIL import Image
import pdfplumber
import os
import sys 

 
def pdf_extract(file_path):
    with pdfplumber.open(file_path) as files:
        string = ''
        for page in files.pages: #for each page in the read pdf file
            pages_text = page.extract_text() or '' #extract texts from each page
            
            for line in pages_text.splitlines(): #splittng the lines in each page with splitlines()
                string += ' '.join(line.split()) + ' ' # .split() further split each line into words and a space, then joining it to the string
            string += '\n' #adding space before new page
    return string.strip()

 
#pdf_extract('/home/abdullahimujaheed/Downloads/Build a Large Language Mode_ (z-library.sk, 1lib.sk, z-lib.sk).pdf')


 
def extract_text_with_ocr(file_path):
    string = ""
    doc = pymupdf.open(file_path)
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

 
def main(file_path):  
    # First try to extract text using pdfplumber
    text = pdf_extract(file_path)
    
    # If no text is found, use OCR
    if not text.strip():
        print("No text found, performing OCR...")
        text = extract_text_with_ocr(file_path)

    print("Extracted Text:")
    print(text)

    # Extract the filename from the URL
    filename = os.path.basename(file_path).replace('.pdf', '.txt')  # Change extension to .txt

    save_text_to_file(text, filename)


 
if __name__ == "__main__":
    file_path= sys.argv[1]  # Get the URL from command line arguments
    main(file_path)