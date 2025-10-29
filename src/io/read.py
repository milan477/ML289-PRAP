from pypdf import PdfReader
from src.config import DATA_DIR
from pathlib import Path
import pandas as pd
from src.schema.dataset import PDF,Page,Dataset

def _read_single_pdf(path: Path = DATA_DIR):
    reader = PdfReader(path)
    print(f"Reading {path.name} ({len(reader.pages)} pages)")

    pdf = PDF(path.name)

    for page in reader.pages:
        pdf.add_page(Page(page.extract_text()))

    return pdf

def read_pdfs(path: Path = DATA_DIR):
    pdfs = []
    for file in path.glob('*.pdf'):
        pdfs.append(_read_single_pdf(file))
    return Dataset(pdfs)

if __name__ == '__main__':
    print(read_pdfs())