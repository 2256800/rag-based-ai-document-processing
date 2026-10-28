import fitz  # PyMuPDF

def detect_file_type(file_path):
    doc = fitz.open(file_path)
    
    text = ""
    for page in doc:
        text += page.get_text()
    
    if len(text.strip()) > 100:
        return "digital_pdf"
    else:
        return "scanned_pdf"

# Example usage
if __name__ == "__main__":
    file_path = "financial_data.pdf"   # replace with your PDF path
    result = detect_file_type(file_path)
    print("File type:", result)
