import os
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.io.download import scrape_database
from src.io.read import read_pdfs
from src.schema.dataset import DocumentType, DocumentDataset
import argparse
from src.config import DATA_DIR

scrape_database(start_page=1, end_page=1)
documents = read_pdfs(DATA_DIR, method='pytesseract', begin = 1, limit = 10)
doc = documents.documents[5]
print(f"Document: {doc.name}")
print(f"Classification: {doc.type}")
doc.show()