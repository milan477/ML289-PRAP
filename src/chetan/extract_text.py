import fitz # PyMuPDF
import os
from pathlib import Path
import json
from tqdm import tqdm

def extract_text_from_pdf(pdf_path):
    # Extract all text from a PDF file
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text.strip()

def process_pdfs(input_dir, output_dir):
    # Process all PDFs in directory and save extracted text
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    pdf_files = list(input_path.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files")

    results = []

    for pdf_file in tqdm(pdf_files, desc="Extracting text"):
        text = extract_text_from_pdf(pdf_file)

        if text:
            result = {
                'filename': pdf_file.name,
                'text': text,
                'char_count': len(text)
            }
            results.append(result)

    output_file = output_path / "extracted_texts.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Extracted text from {len(results)} PDFs")
    print(f"Saved to {output_file}")

    return results

if __name__ == "__main__":
    input_dir = "data/records"
    output_dir = "data/processed"

    results = process_pdfs(input_dir, output_dir)
