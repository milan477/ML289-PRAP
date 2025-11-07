from bertopic import BERTopic
from src.schema.dataset import Dataset, Document
import spacy
from spacy.cli import download
from transformers import pipeline
import numpy as np

def get_text_features(dataset):
    print(np.random.randint(0,100))
    features = []
    for document in dataset.documents:
        fv = {}
        # fv.update(_get_length(document))
        fv.update(_get_names_transformers(document))
        features.append(fv)

    # _get_topics(dataset.documents)
    return features

def _get_names_spacy(document: Document):
    try:
        model = spacy.load("en_core_web_sm")
    except OSError:
        download("en_core_web_sm")
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
