from dataclasses import dataclass
import pandas as pd
from typing_extensions import override

import torch
import os
import pytesseract
import pymupdf

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pdf2image import convert_from_path
from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageDraw
from PIL.PpmImagePlugin import PpmImageFile
from dataclasses import dataclass

from io import BytesIO

from src.preprocessing.preprocess import _clean_text


@dataclass
class Page:
    text: str
    ocr_data: pd.DataFrame
    image: PpmImageFile

    def __str__(self):
        return self.text

    def __len__(self):
        return len(self.text)

    def draw_boxes(self):
        coordinates = self.ocr_data[['left', 'top', 'width', 'height']]
        actual_boxes = []
        for idx, row in coordinates.iterrows():
            x, y, w, h = tuple(row) # the row comes in (left, top, width, height) format
            actual_box = [x, y, x+w, y+h] # we turn it into (left, top, left+width, top+height) to get the actual box
            actual_boxes.append(actual_box)

        image = self.image.copy()
        draw = ImageDraw.Draw(image, "RGB")
        for box in actual_boxes:
            draw.rectangle(box, outline='red')
        return image

@dataclass
class Document:
    name: str
    location: Path
    pages: list[Page]
    format: str = "pdf"

    def add_page(self,page):
        self.pages.append(page)

    def __str__(self):
        return self.name

    def __repr__(self):
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

    def inspect(self,pagenr=0):
        assert pagenr < len(self.pages)
        return self.pages[pagenr].draw_boxes()

    def get_full_text(self):
        content = ''
        for page in self.pages:
            content += page.text
        return content

    def read(self):
        print(self.get_full_text())

    def is_image(self,threshold = 200):

        clean_text = _clean_text(self.get_full_text())
        words = 0
        for word in clean_text.split():
            if len(word) > 2:
                words += 1


        if len(clean_text) < threshold or words < 20:
            return True
        return False

    def show(self, ax=None, label=None, idx=None):
        if not ax:
            show_image = True
            fig, ax = plt.subplots(figsize=(8,10))
        else:
            show_image = False

        ax.imshow(self.pages[0].image)
        ax.axis('off')
        id = 'id: ' + str(self.get_id())
        annotation = ' (image: ' + str(idx) + ')' if idx is not None else ''
        title = id + annotation + label if label else id + annotation

        ax.set_title(title)

        if show_image:
            plt.show()

    def get_id(self):
        return int(self.name.split('id')[1].split('.pdf')[0])

    def extract_images(self):
        doc = pymupdf.open(self.location)
        image_list = []
        pages = []
        for pg_idx in range(len(doc)):
            page = doc[pg_idx]
            images = page.get_images()
            image_list.extend(images)
            pages.extend([pg_idx] * len(images))


        num_images = len(image_list)
        width = min(num_images, 4)
        height = 1 + num_images % width

        fig = plt.figure(figsize=(width*4, height*4))

        for index, image in enumerate(image_list):

            xref = image[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(BytesIO(image_bytes))

            if np.array(image).mean() < 1:
                continue

            ax = fig.add_subplot(height*4, width*4, index + 1, title=f"Page {pages[index]+1}")
            ax.axis('off')

            ax.imshow(image)

        plt.tight_layout()
        plt.show()

@dataclass
class DocumentDataset:
    documents : list[Document]

    def info(self):
        return {'size': len(self.documents)}

    def remove_images(self):
        trimmed = []
        for document in self.documents:
            if not document.is_image():
                trimmed.append(document)
        return DocumentDataset(trimmed)

    def __getitem__(self,key):
        return self.documents[key]

    def show(self, start_index = 0, end_index = None, indices = None):

        if indices:
            num_images = len(indices)
        else:
            if not end_index:
                if not start_index:
                    start_index = 0
                    end_index = min(100, len(self.documents))
                else:
                    end_index = start_index + 1

            indices = range(start_index, end_index)
            num_images = len(indices)

        width = min(num_images, 4)
        height = int(np.ceil(num_images / width))

        fig = plt.figure(figsize=(width*4, height*4))

        for i, index in enumerate(indices):
            ax = fig.add_subplot(height, width, i + 1)
            document = self.documents[index]
            document.show(ax, idx=index)

        plt.tight_layout()
        plt.show()

    def read(self, index):
        document = self.documents[index]
        document.read()

    def inspect(self, index, pagenr=0):
        document = self.documents[index]
        return document.inspect(pagenr)

    def extract_images(self, index):
        document = self.documents[index]
        document.extract_images()

    def show_clusters(self, labels):
        for label in set(labels):
            print(f"{'_'*100}\nCluster {label}:")
            indices = []
            for i, doc in enumerate(self.documents):
                if labels[i] == label:
                    indices.append(i)
            self.show(indices=indices)
