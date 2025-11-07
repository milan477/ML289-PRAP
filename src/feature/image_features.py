import pymupdf4llm
import pymupdf
import pandas as pd


def _get_single_image_count(document):
  doc = pymupdf.open(document.location)
  # print(f"Analyzing {file.name}")
  for pg_idx in range(len(doc)):
    page = doc[pg_idx]
    image_list = page.get_images()
    # if image_list:
    #   print(f"Found {len(image_list)} images on page {pg_idx}")
    # else:
    #   print("No images found on page", pg_idx)
  # print(f"Found {len(image_list)} images")
  return {'image count': len(image_list)}

def get_image_features(dataset):
    features = []
    for document in dataset.documents:
        fv = {}
        fv.update(_get_single_image_count(document))
        features.append(fv)
    return features