from dataclasses import dataclass

import pandas as pd

@dataclass
class Page:
    text: str
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

@dataclass
class Document:
    name: str
    format: str

    def __str__(self):
        return 'self'

    def describe(self):
        return {"name": self.name, "format": self.format}

@dataclass
class PDF (Document):
    {}
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

    def describe(self):
        return {'type': self.type, 'name': self.name, 'format': self.format, 'pages': "".join(str(self.pages))}



@dataclass
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

    def __len__(self):
        return len(self.documents)

    def to_frame(self):
        all_documents = []
        for document in self.documents:
            all_documents.append(document.describe())
        return pd.DataFrame(all_documents)



