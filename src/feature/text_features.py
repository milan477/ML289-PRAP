from bertopic import BERTopic
import torch
from sklearn.preprocessing import StandardScaler

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
import builtins


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


def ask_a_question(question = "what kind of document is this?", ocrs=[], model_for_QA=None, tokenizer_for_QA=None):
    types = []
    for ocr_data in tqdm(ocrs):
        words, boxes = convert_text_to_boxes(ocr_data)

        encoding = tokenizer_for_QA(
            question.split(),
            words,
            is_split_into_words=True,
            return_token_type_ids=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )
        encoding["bbox"] = torch.tensor([boxes])

        bbox = []
        for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
            if s == 1 and w is not None:
                bbox.append(boxes[w])
            elif i == tokenizer_for_QA.sep_token_id:
                bbox.append([1000] * 4)
            else:
                bbox.append([0] * 4)
        encoding["bbox"] = torch.tensor([bbox])

        outputs = model_for_QA(**encoding)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        start_probabilities = torch.softmax(start_scores, dim=1).squeeze()
        end_probabilities = torch.softmax(end_scores, dim=1).squeeze()

        topk_start = torch.topk(start_probabilities, k=3)[1].squeeze()
        topk_end = torch.topk(end_probabilities, k=3)[1].squeeze()

        word_ids = encoding.word_ids(0)
        possible_answers = []
        probabilities = []
        for start_token, end_token in zip(topk_start, topk_end):
            if start_probabilities[start_token] > 0.1 and end_probabilities[end_token] > 0.1:
                start_word, end_word = word_ids[start_token], word_ids[end_token]

                if start_word is not None and end_word is not None and start_word <= end_word and end_word < start_word + 10:
                    possible_answers.append(" ".join(words[start_word : end_word + 1]))
                    probabilities.append([start_probabilities[start_token], end_probabilities[end_token]])

        types.append([possible_answers if len(possible_answers) > 0 else ["N/A"], probabilities])

    return types


def convert_text_to_boxes(ocr_data: pd.DataFrame):
    words = []
    boxes = []

    page_sizes = ocr_data[['width', 'height']].max().values
    w, h = page_sizes[0], page_sizes[1]

    for i, text in enumerate(ocr_data['text']):

        if text.strip() == "" and ocr_data['level'][i] != 5:
            continue
        words.append(text)


        x0 = builtins.int(ocr_data['left'][i] / w * 1000)
        y0 = builtins.int(ocr_data['top'][i] / h * 1000)
        x1 = builtins.int((ocr_data['left'][i] + ocr_data['width'][i]) / w * 1000)
        y1 = builtins.int((ocr_data['top'][i] + ocr_data['height'][i]) / h * 1000)

        boxes.append([x0, y0, x1, y1])

    # print(f'Converted OCR data into {len(words)} words and {len(boxes)} boxes')
    return words, boxes



class EmbeddingExtractor(FeatureExtractor):

    def __init__(self, tokenizer, model):
        super().__init__()
        self.tokenizer = tokenizer
        self.model = model


    def extract(self, dataset: DocumentDataset):
        scaler = StandardScaler()

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

            pd.DataFrame(encoding).head()

            bbox = []
            for i, s, w in zip(encoding.input_ids[0], encoding.sequence_ids(0), encoding.word_ids(0)):
                if s == 1 and w is not None:
                    bbox.append(boxes[w])
                elif i == self.tokenizer.sep_token_id:
                    bbox.append([1000] * 4)
                else:
                    bbox.append([0] * 4)

            encoding["bbox"] = torch.tensor([bbox])
            with torch.no_grad():
                outputs = self.model(**encoding)

            embedding = outputs.last_hidden_state.detach().numpy().squeeze()

            mean_embedding = embedding[encoding['attention_mask'].squeeze()].mean(axis=0)

            doc_embedding = outputs.last_hidden_state[:, 0, :]

            no_mask_length = encoding['attention_mask'].squeeze().sum().item()

            embeddings.append(doc_embedding.detach().numpy().squeeze())

        result = np.array(embeddings).squeeze()
        if scaler:
            return scaler.fit_transform(result)

        return



class Embedder(abc.ABC):
    """Base class for image embedders."""

    @abc.abstractmethod
    def embed(self, images: np.array):
        pass




class DinoTransformer(Embedder):
    """DINOv2 transformer for image embedding extraction."""

    def __init__(self):
        dinov2_vits14 = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        self.device = get_device()
        self.dino = dinov2_vits14.to(self.device)
        self.name = "dino_v2_vits14"

    def get_embedding(self, image):
        embedding = self.dino(image.to(self.device))
        return np.array(embedding[0].cpu().numpy()).reshape(1, -1).tolist()

    def embed(self, images: np.array, filename=None):
        # TODO: apply multiworker
        tensor = self._to_trainable_tensor(images)

        all_embeddings = {}

        with torch.no_grad():
            for i, image in enumerate(tqdm(tensor)):
                embedding = self.get_embedding(image)
                all_embeddings["image" + str(i)] = embedding

        return all_embeddings

    def _to_trainable_tensor(self, images: np.array):
        # normalize images to [0,1]
        images = images.astype("float32") / 255.0

        # convert to tensor
        tensor = torch.from_numpy(images)

        # permute to (N, C, H, W)
        tensor = tensor.permute(0, 3, 1, 2)

        # standardize according to ImageNet's mean and std
        transform = trans.Compose(
            [trans.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )

        # make image size a multiple of dino's patch size
        patch_size = self.dino.patch_size
        H = tensor.shape[2]
        W = tensor.shape[3]
        resulting_tensor = (
            tensor[:, :3, : (H - H % patch_size), : (W - W % patch_size)]
            .unsqueeze(1)
            .float()
        )

        print(
            f"converted {len(images)} images to trainable tensor of shape {resulting_tensor.shape} for {self.name}"
        )
        return transform(resulting_tensor).to(self.device)



















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