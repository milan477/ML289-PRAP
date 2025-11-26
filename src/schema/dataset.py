from dataclasses import dataclass
import pathlib
import pandas as pd
from typing_extensions import override


@dataclass
class Page:
    text: str
    def __init__(self, text: str):
        self.text = text

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)

@dataclass
class Document:
    name: str
    format: str
    location: pathlib.Path

    def __str__(self):
        return self.name

    def describe(self):
        return {"name": self.name, "format": self.format}

    def get_full_content(self):
        pass

    def __len__(self):
        pass

    def set_tokens(self, tokens):
        pass

    def get_tokens(self):
        return []


@dataclass
class PDF (Document):
    pages: list[Page]
    type: str
    tokens: list[str]

    def __init__(self,name,location):
        self.name = name
        self.format = "pdf"
        self.pages = []
        self.type = "unknown"
        self.location = location

    def add_page(self,page):
        self.pages.append(page)

    def __str__(self):
        full_text = f"\n########################################################## begin of {self.name} of type {self.type} ##########################################################\n"
        full_text += self.get_full_content()
        full_text += f"\n########################################################## end of {self.name} of type {self.type} ##########################################################\n"
        return full_text

    def describe(self):
        return {'type': self.type, 'name': self.name, 'format': self.format, 'pages': "".join(str(self.pages))}

    @override
    def get_full_content(self):
        full_text = ""
        for page in self.pages:
            full_text += str(page)
        return full_text

    def __len__(self):
        return len(self.pages)

    def set_tokens(self, tokens):
        print('adding tokens')
        self.tokens = tokens

    def get_tokens(self):
        return self.tokens

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



