from bertopic import BERTopic

from src.schema.dataset import DocumentDataset, Document
from transformers import pipeline
import numpy as np
from langid import classify
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from src.preprocessing.preprocess import _clean_text

import abc
from abc import abstractmethod

from tqdm import tqdm
import torch

from typing_extensions import override


class FeatureExtractor(abc.ABC):
    @abstractmethod
    def extract(self, dataset: DocumentDataset):
        pass


class BagOfWordsExtractor(FeatureExtractor):

    @override
    def extract(self, dataset: DocumentDataset, vocabulary):
        vectorizer = CountVectorizer(
            vocabulary=vocabulary,
            ngram_range=(1,4),
            lowercase=True,
            token_pattern=r'\b[\w:]+'
            )

        contents = [_clean_text(document.get_full_text()) for document in dataset.documents]

        X = vectorizer.fit_transform(np.array(contents).squeeze())

        return X.toarray()


class TFIDFExtractor(FeatureExtractor):

    @override
    def extract(self, dataset: DocumentDataset, vocabulary):
        vectorizer = TfidfVectorizer(
            vocabulary=vocabulary,
            ngram_range=(1,4),
            lowercase=True,
            token_pattern=r'\b[\w:]+'
            )

        contents = [_clean_text(document.get_full_text()) for document in dataset.documents]

        X = vectorizer.fit_transform(np.array(contents).squeeze())

        return X.toarray()


def convert_text_to_boxes(ocr_data: pd.DataFrame):
    words = []
    boxes = []

    page_sizes = ocr_data[['width', 'height']].max().values
    w, h = page_sizes[0], page_sizes[1]

    for i, text in enumerate(ocr_data['text']):

        if text.strip() == "" and ocr_data['level'][i] != 5:
            continue
        words.append(text)
        x0 = int(ocr_data['left'][i] / w * 1000)
        y0 = int(ocr_data['top'][i] / h * 1000)
        x1 = int((ocr_data['left'][i] + ocr_data['width'][i]) / w * 1000)
        y1 = int((ocr_data['top'][i] + ocr_data['height'][i]) / h * 1000)
        boxes.append([x0, y0, x1, y1])

    # print(f'Converted OCR data into {len(words)} words and {len(boxes)} boxes')
    return words, boxes


class EmbeddingExtractor(FeatureExtractor):

    tokenizer = None
    model = None

    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model

    @override
    def extract(self, dataset: DocumentDataset):
        embeddings = []

        ocrs = [document.pages[0].ocr_data for document in dataset.documents]

        for ocr_data in ocrs:
            words, boxes = convert_text_to_boxes(ocr_data)

            encoding = self.tokenizer(
                words,
                boxes=boxes,
                return_token_type_ids=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
            )
            encoding["bbox"] = torch.tensor([boxes])
            pd.DataFrame(encoding).head()

            bbox = []
            for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
                if s == 1 and w is not None:
                    bbox.append(boxes[w])
                elif i == tokenizer.sep_token_id:
                    bbox.append([1000] * 4)
                else:
                    bbox.append([0] * 4)
            encoding["bbox"] = torch.tensor([bbox])
            with torch.no_grad():
                outputs = self.model(**encoding)

            embedding = outputs.last_hidden_state.detach().numpy().squeeze()

            mean_embedding = embedding[encoding['attention_mask'].squeeze()].mean(axis=0)
            no_mask_length = encoding['attention_mask'].squeeze().sum().item()

            embeddings.append(mean_embedding)

        return np.array(embeddings).squeeze()





















# def get_text_features(dataset):
#     features = []
#     for document in dataset.documents:
#         fv = {}
#         fv.update(_get_character_length(document))
#         fv.update(_get_language(document))
#         fv.update(_get_top_words(document))
#         fv.update(_get_names_spacy(document))
#         features.append(fv)

#     _get_topics(dataset.documents)
#     print(_get_top_words_tfidf(dataset))

#     return features

# def _get_names_spacy(document: PDF):
#     names = [ent.text for ent in document.tokens.ents if ent.label_ == "PERSON"]
#     return {'names': names}

# def _get_names_transformers(document: Document):
#     model = pipeline('ner',
#                      grouped_entities=True,
#                      model="dbmdz/bert-large-cased-finetuned-conll03-english",
#                      tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english"
#                      )

#     ents = model(document.get_full_content())
#     names = [ent['word'] for ent in ents if ent['entity_group'] == "PER"]
#     return {'names': names}

# def _get_character_length(document: Document, tokenize=False):
#     count = 0
#     if tokenize:
#         raise NotImplementedError

#     if document.format == 'pdf':
#         for page in document.pages:
#             count += len(page)

#     return {"character length": count}

# def _get_topics(documents: list[PDF]):
#     model = BERTopic(verbose=True)

#     topics, probs = model.fit_transform([" ".join(_remove_stopwords(document)) for document in documents])

#     print(topics)
#     print(model.get_topic_info())
#     print(model.get_topic(0))

# def _get_language(document: Document):
#     return {'language': classify(document.get_full_content())[0]}

# def _get_summary(document: Document):
#     return {'summary': 'summary'}

# def _remove_stopwords(document: PDF):
#     return [token.lemma_ for token in document.get_tokens() if not token.is_stop]

# def _get_top_words(document: PDF):
#     tokens_without_stopwords = _remove_stopwords(document.get_tokens())
#     s = pd.Series(tokens_without_stopwords).value_counts(ascending=False)
#     i = 0
#     top_words = []
#     for word in s.index:
#         if not re.findall(r"\d", word):
#             top_words.append(word)
#         if len(top_words) >= 10:
#             break
#     return {'top words':top_words}

# def _get_top_words_tfidf(dataset: Dataset):
#     texts = [" ".join(_remove_stopwords(document.get_tokens())) for document in dataset.documents]

#     vectorizer = TfidfVectorizer(stop_words="english")
#     X = vectorizer.fit_transform(texts)

#     feature_names = np.array(vectorizer.get_feature_names_out())

#     top_words = []
#     for i in range(X.shape[0]):
#         row = X[i].toarray().ravel()
#         top_indices = row.argsort()[-10:][::-1]
#         top_words.append(feature_names[top_indices])

#     return top_words