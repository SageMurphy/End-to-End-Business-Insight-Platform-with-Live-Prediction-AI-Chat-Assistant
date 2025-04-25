import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file."""
    text = ""
    try:
        pdf_document = fitz.open(pdf_path)
        for page_number in range(pdf_document.page_count):
            page = pdf_document.load_page(page_number)
            text += page.get_text()
        pdf_document.close()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_text_from_multiple_pdfs(pdf_folder):
    """Extracts text from all PDF files in a folder."""
    all_texts = {}
    import os
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            text = extract_text_from_pdf(pdf_path)
            all_texts[filename] = text
    return all_texts

if __name__ == "__main__":
    pdf_folder = 'docs'  # Replace with the actual path to your PDF folder
    extracted_texts = extract_text_from_multiple_pdfs(pdf_folder)
    for filename, text in extracted_texts.items():
        print(f"Extracted text from {filename} (first 200 characters):\n{text[:200]}...\n")
        # You might want to save these texts to a file for inspection