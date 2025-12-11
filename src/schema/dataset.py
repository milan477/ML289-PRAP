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

    def is_image(self,threshold = 10):
        if len(self.get_full_text()) < threshold:
            return True
        return False

    def show(self, ax, label=None):
        ax.imshow(self.pages[0].image)
        ax.axis('off')
        title = 'id: ' + str(self.get_id()) + label if label else 'id: ' + str(self.get_id())
        ax.set_title(title)

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

    def show(self, start_index = 0, end_index = None):
        if not end_index:
            end_index = start_index + 1
        num_images = end_index-start_index
        width = min(num_images, 4)
        height = 1 + num_images % width

        fig = plt.figure(figsize=(width*4, height*4))

        for i, index in enumerate(range(start_index, end_index)):
            ax = fig.add_subplot(height, width, i + 1)
            document = self.documents[index]
            document.show(ax)

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
