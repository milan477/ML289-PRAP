
from PIL import Image
Image.MAX_IMAGE_PIXELS = 300000000  # to avoid DecompressionBombError for large images

from pypdf import PdfReader
from src.config import DATA_DIR
from pathlib import Path
import pandas as pd
from src.schema.dataset import Page,DocumentDataset, Document
from pdf2image import convert_from_path
import pytesseract
from src.types.document_type import DocumentType



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

def classify_by_content(text: str) -> DocumentType:
    """
    Classify document based on OCR'd text content
    Args:
        text: Text extracted from the document
    Returns:
        DocumentType enum
    """
    text_lower = text.lower()
    
    # Check first 1000 characters for most indicators
    header = text_lower[:1000]
    
    # Also check full text for certain keywords
    full_text = text_lower
    
    print(f"  Analyzing content: {header[:100]}...")
    
    # DISCOVERY PACKAGE - Most important for police records
    discovery_keywords = [
        'discovery package'
    ]
    if any(keyword in header for keyword in discovery_keywords):
        return DocumentType.DISCOVERY_PACKAGE
    
    # COMMISSION AGENDA
    agenda_keywords = [
        'commission agenda', 'meeting agenda', 'board agenda'
    ]
    if any(keyword in header for keyword in agenda_keywords):
        return DocumentType.COMMISSION_AGENDA
    
    # PRESS RELEASE
    press_keywords = [
        'for immediate release', 'press release', 'media advisory'
    ]
    if any(keyword in header for keyword in press_keywords):
        return DocumentType.PRESS_RELEASE
    
    # CORRESPONDENCE
    correspondence_keywords = [
        'memorandum', 'memo to', 'memo from', 're:', 'regarding:',
        'dear chief', 'dear commissioner', 'sincerely', 'cc:',
        'subject:', 'letter to', 'letter from', 'correspondence'
    ]
    if any(keyword in header for keyword in correspondence_keywords):
        return DocumentType.CORRESPONDENCE
    
    # REPORTS (annual, statistical, etc.)
    report_keywords = [
        'incident report', "coroner report", 'death in custody report', 'investigative report'
    ]
    if any(keyword in header for keyword in report_keywords):
        return DocumentType.REPORTS
    
    return DocumentType.UNKNOWN

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

    # Classify based on OCR'd text content from first page
    if pages and pages[0].text:
        print("  Classifying by content...")
        doc_type = classify_by_content(pages[0].text)
        print(f"  Classified as: {doc_type.value}")
    else:
        print("  No text found - classifying as UNKNOWN")
        doc_type = DocumentType.UNKNOWN
    
    return Document(
        pages=pages, 
        name=path.name, 
        format='pdf', 
        location=path,
        type=doc_type
    )



def read_pdfs(path: Path, method='pytesseract', begin=1, limit=10, 
              filter_types: list[DocumentType] = None) -> DocumentDataset:
    """
    Read PDFs and optionally filter by document type
    
    Args:
        path: Directory containing PDFs
        method: OCR method to use
        begin: Starting file ID
        limit: Number of files to read
        filter_types: Optional list of DocumentType enums to include (None = all)
    
    Example:
        # Only load discovery packages and reports
        dataset = read_pdfs(
            path=Path("data/policerecords/pdfs"),
            limit=100,
            filter_types=[DocumentType.DISCOVERY_PACKAGE, DocumentType.REPORTS]
        )
    """
    count = 0
    pdfs = []
    files = {get_file_id(f.name): f for f in path.iterdir() 
             if f.is_file() and f.name.endswith('.pdf')}

    if method == 'pytesseract':
        for filenr in range(begin, limit+1):
            if filenr not in files:
                print(f'File id{filenr}.pdf not found, skipping...')
                continue
            
            file = files[filenr]
            doc = _read_single_pdf_pytesseract(file)
            
            # Filter by type if specified
            if filter_types:
                if doc.type not in filter_types:
                    print(f'  ⊘ Skipping {file.name} - type {doc.type} not in filter')
                    continue
            
            pdfs.append(doc)
            count += 1
            print(f'Reading pdf {count}/{limit} - Type: {doc.type}')

            if count == limit:
                break
    else:
        raise ValueError(f"Unknown method {method} for reading PDF")
    
    return DocumentDataset(pdfs)