
import spacy
import re

# def preprocess(dataset):
#     for document in dataset.documents:
#         document.set_tokens(_tokenize_spacy(document))

def _clean_text(text):
    text = text.lower()
    text = re.sub(r"[^A-Za-z\s]", "", text)
    text = re.sub(r'\s',' ', text)
    return text

# def _tokenize_spacy(document: Document):
#     try:
#         model = spacy.load("en_core_web_sm")
#     except OSError:
#         spacy.cli.download("en_core_web_sm")
#         model = spacy.load("en_core_web_sm")

#     text = _clean_text(document.get_full_content())
#     doc = model(text)
#     return [token for token in doc]