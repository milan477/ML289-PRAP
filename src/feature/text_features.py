from bertopic import BERTopic
from src.schema.dataset import Dataset, Document
import spacy
from spacy.cli import download
from transformers import pipeline
import numpy as np
from langid import classify
# from textblob import Text
import pandas as pd
import re

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer

def get_text_features(dataset):
    print(101)
    features = []
    for document in dataset.documents:
        fv = {}
        # fv.update(_get_length(document))
        # fv.update(_get_language(document))
        fv.update(_get_top_words(document))
        # fv.update(_get_names_spacy(document))
        features.append(fv)

    # _get_topics(dataset.documents)

    print(_get_top_words_tfidf(dataset))
    return features

def _get_names_spacy(document: Document):
    try:
        model = spacy.load("en_core_web_sm")
    except OSError:
        spacy.cli.download("en_core_web_sm")
        model = spacy.load("en_core_web_sm")

    doc = model(document.get_full_content())
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    return {'names': names}

def _get_names_transformers(document: Document):
    model = pipeline('ner',
                     grouped_entities=True,
                     model="dbmdz/bert-large-cased-finetuned-conll03-english",
                     tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english"
                     )

    ents = model(document.get_full_content())
    names = [ent['word'] for ent in ents if ent['entity_group'] == "PER"]
    return {'names': names}


def _get_length(document: Document,tokenize=False):
    count = 0
    if tokenize:
        raise NotImplementedError

    if document.format == 'pdf':
        for page in document.pages:
            count += len(page)

    return {"character length": count}


def _get_topics(documents: list[Document]):
    model = BERTopic(verbose=True)
    topics, probs = model.fit_transform([document.get_full_content() for document in documents])

    print(topics)
    print(model.get_topic_info())
    print(model.get_topic(0))


def _get_language(document: Document):
    return {'language': classify(document.get_full_content())[0]}

def _get_summary(document: Document):
    return {'summary': 'summary'}

def _get_top_words(document: Document):
    text = _remove_stopwords(document)
    s = pd.Series(text).value_counts(ascending=False)
    i = 0
    top_words = []
    for word in s.index:
        if not re.findall(r"\d", word):
            top_words.append(word)
        if len(top_words) >= 10:
            break
    return {'top words':top_words}

def _remove_stopwords(document: Document):
    return [word for word in document.get_full_content().split() if word.lower() not in ENGLISH_STOP_WORDS]

def _remove_numbers(words : list[str]):
    return [word for word in words if len(re.findall(r'\d')) > 0]

def _get_top_words_tfidf(dataset: Dataset):
    texts = [document.get_full_content() for document in dataset.documents]

    vectorizer = TfidfVectorizer(stop_words="english")
    X = vectorizer.fit_transform(texts)
    print(X)

    feature_names = np.array(vectorizer.get_feature_names_out())
    print(feature_names)

    top_words = []
    for i in range(X.shape[0]):
        row = X[i].toarray().ravel()
        top_indices = row.argsort()[-10:][::-1]
        top_words.append(feature_names[top_indices])

    return top_words


def _tokenize(document: Document):
    text = document.get_full_content()