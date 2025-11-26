from pypdf import PdfReader
from src.config import DATA_DIR
from pathlib import Path
import pandas as pd
from src.schema.dataset import PDF,Page,Dataset
from pdf2image import convert_from_path
import pytesseract

def _read_single_pdf_pypdf(path: Path) -> PDF:
    reader = PdfReader(path)
    print(f"Reading {path.name} ({len(reader.pages)} pages)")

    pdf = PDF(name=path.name, location=path)

    for page in reader.pages:
        pdf.add_page(Page(page.extract_text()))

    return pdf

def _read_single_pdf_pytesseract(path: Path) -> PDF:
    print(f"Reading {path.name} with pytesseract")
    pages = convert_from_path(path)

    page = pages[0]
    data = pytesseract.image_to_data(page,output_type=pytesseract.Output.DICT)


    return page, data

def _read_single_pdf(path: Path, method: str = 'pypdf'):
    if method == 'pypdf':
        return _read_single_pdf_pypdf(path)

    if method == 'pytesseract':
        return _read_single_pdf_pytesseract(path)
    else:
        raise ValueError(f"Unknown method {method} for reading PDFs")

        # for file in path.glob('*.pdf'):
    #     pdfs.append(_read_single_pdf(file))
    #     if i == 1: break
    #     i += 1
def read_pdfs(path: Path = DATA_DIR, method: str = 'pypdf') -> Dataset:

    pdfs = []
    print(f"Reading {path.name} ({len(pdfs)} pages) using {method}")
    i = 0
    for i in range(2):
        pdfs.append(_read_single_pdf(path / f"record{i}.pdf", method=method))
    return Dataset(pdfs)
if __name__ == '__main__':
    print(f'read {len(read_pdfs())} PDFs')