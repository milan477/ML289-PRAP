from pypdf import PdfReader
from src.config import DATA_DIR
from pathlib import Path
import pandas as pd
from src.schema.dataset import PDF,Page,Dataset

def _read_single_pdf(path):
    reader = PdfReader(path)
    print(f"Reading {path.name} ({len(reader.pages)} pages)")

    pdf = PDF(name=path.name, location=path)

    for page in reader.pages:
        pdf.add_page(Page(page.extract_text()))

    return pdf

def read_pdfs(path: Path = DATA_DIR):
    pdfs = []
    i = 0
    for i in range(100):
        pdfs.append(_read_single_pdf(path / f"record{i}.pdf"))

    # for file in path.glob('*.pdf'):
    #     pdfs.append(_read_single_pdf(file))
    #     if i == 1: break
    #     i += 1
    return Dataset(pdfs)

if __name__ == '__main__':
    print(len(read_pdfs()))