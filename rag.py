import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import os
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from PIL import Image
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# --- CONFIGURATION ---
PDF_PATH = "financial_data.pdf"
OUTPUT_DIR = "./output_images"
DB_DIR = "./vector_db"
os.makedirs(OUTPUT_DIR, exist_ok=True)

class RAGPipeline:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.image_paths = []
        self.structured_content = {"tables": [], "text": [], "visuals": []}

    def step_1_and_2_render_images(self, dpi=300):
        """Flattening and rendering pages to 300 DPI images."""
        print("Rendering PDF to images...")
        doc = fitz.open(self.pdf_path)
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            zoom = dpi / 72
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
            img_path = os.path.join(OUTPUT_DIR, f"page_{page_num}.png")
            pix.save(img_path)
            self.image_paths.append(img_path)
        return self.image_paths

    def step_3_extract_content(self):
        """Precise extraction of tables, text via OCR, and image captions."""
        print("Extracting layout-aware content...")
        
        # 1. Extract Tables using pdfplumber
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    md_table = "\n".join([" | ".join([str(c) if c else "" for c in r]) for r in table])
                    self.structured_content["tables"].append(md_table)

        # 2. Extract Text via OCR (Tesseract)
        for img_path in self.image_paths:
            text = pytesseract.image_to_string(Image.open(img_path))
            self.structured_content["text"].append(text)

        # 3. Visual Context (Mock Vision LLM logic)
        self.structured_content["visuals"].append("Visual context: Technical diagram of data architecture.")

    def step_4_chunking(self):
        """Layout-aware chunking to preserve context."""
        print("Creating layout-aware chunks...")
        chunks = []
        for category, items in self.structured_content.items():
            for item in items:
                if len(item.strip()) > 20:
                    chunks.append(f"[{category.upper()}]\n{item}")
        return chunks

    def step_5_and_6_vectorize(self, chunks):
        """Generating embeddings and storing in Vector Database."""
        print("Vectorizing and storing in ChromaDB...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            persist_directory=DB_DIR
        )
        return vectordb

# --- EXECUTION ---
if __name__ == "__main__":
    # Initialize the pipeline
    pipeline = RAGPipeline(PDF_PATH)
    
    # Run Step 1 & 2: Flattening & Rendering 
    pipeline.step_1_and_2_render_images()
    
    # Run Step 3: Extraction (Tables, OCR, Vision LLM) 
    pipeline.step_3_extract_content()
    
    # Run Step 4: Layout-Aware Chunking 
    content_chunks = pipeline.step_4_chunking()
    
    # Run Step 5 & 6: Vector Embeddings & Database 
    db = pipeline.step_5_and_6_vectorize(content_chunks)
    
    print("\n--- Pipeline Complete ---")
    print(f"Stored in: {DB_DIR}")
    