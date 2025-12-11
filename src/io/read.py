
from PIL import Image
Image.MAX_IMAGE_PIXELS = 300000000  # to avoid DecompressionBombError for large images

from pypdf import PdfReader
from src.config import DATA_DIR
from pathlib import Path
import pandas as pd
from src.schema.dataset import Page,DocumentDataset, Document
from pdf2image import convert_from_path
import pytesseract


def concatenate(s):
    return ' '.join([str(t) for t in s if str(t) != 'nan' and str(t).strip() != ''])

def get_text(ocr_data, show=False):
    lines = ocr_data.groupby(
        ['page_num','block_num','line_num'])[['text']].agg(concatenate)
    text = str('\n'.join(lines['text'].values))
    if show:
       print(text)
    return text

def get_file_id(filename):
    return int(filename.split('id')[1].split('.pdf')[0])

def _read_single_pdf_pytesseract(path: Path) -> Document:
    print(f"Reading {path.name} with pytesseract")

    images = convert_from_path(path, first_page=1, last_page=1, dpi=350)
    pages = []

    for image in images[:1]:
        print(f'Processing page {len(pages)+1} / {len(images)}')

        ocr_data = pd.DataFrame(
                pytesseract.image_to_data(
                    image = image,
                    lang = 'eng',
                    output_type=pytesseract.Output.DICT)
                    )
        text = get_text(ocr_data, show=False)
        pages.append(Page(text=text, ocr_data = ocr_data, image = image))

        # image = preprocess(page).unsqueeze(0).to(device)

    print(' --> done')
    return Document(pages = pages, name = path.name, format = 'pdf', location = path)



def read_pdfs(path: Path, method = 'pytesseract', begin = 1, limit = 10) -> DocumentDataset:
    count = 0
    pdfs = []
    files = {get_file_id(f.name) : f for f in path.iterdir() if f.is_file() and f.name.endswith('.pdf')}

    if method == 'pytesseract':
        for filenr in range(begin,limit+1):
            if filenr not in files:
                print(f'File id{filenr}.pdf not found, skipping...')
                continue
            file = files[filenr]
            pdfs.append(_read_single_pdf_pytesseract(file))
            count += 1
            print(f'reading pdf {count}/{limit}')

            if count == limit:
                break
    else:
        raise ValueError(f"Unknown method {method} for reading PDF")
    return DocumentDataset(pdfs)