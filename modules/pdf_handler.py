import fitz  # PyMuPDF

class PDFHandler:
    @staticmethod
    def extract_text(pdf_path):
        doc = fitz.open(pdf_path)
        return "\n".join(page.get_text() for page in doc)
