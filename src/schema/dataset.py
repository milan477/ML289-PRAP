from dataclasses import dataclass

class Page:
    text: str
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

class Document:
    name: str
    format: str

    def __str__(self):
        return 'self'

@dataclass
class PDF (Document):
    pages: list[Page]
    type: str

    def __init__(self,name):
        self.name = name
        self.format = "pdf"
        self.pages = []
        self.type = "unknown"

    def add_page(self,page):
        self.pages.append(page)

    def __str__(self):
        full_text = f"\n########################################################## begin of {self.name} of type {self.type} ##########################################################\n"
        for page in self.pages:
            full_text += str(page)
        full_text += f"\n########################################################## end of {self.name} of type {self.type} ##########################################################\n"
        return full_text

class Dataset:
    name: str
    documents: list[Document]

    def __init__(self,documents):
        self.documents = documents

    def __str__(self):
        full_dataset = ""
        for document in self.documents:
            full_dataset += str(document)
        return full_dataset