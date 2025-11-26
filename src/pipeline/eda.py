#%%
from sklearn.linear_model import LogisticRegression


def reload_all():
    import importlib

    import src.io.read as read
    importlib.reload(read)

    import src.schema.dataset as dataset
    importlib.reload(dataset)

    import src.feature.text_features as text_features
    importlib.reload(text_features)

    import src.feature.image_features as image_features
    importlib.reload(image_features)

    import src.preprocessing.preprocess as preprocess
    importlib.reload(preprocess)

reload_all()

from src.io.read import read_pdfs

from src.schema.dataset import Dataset

from src.feature.image_features import (
    get_image_features
)

from src.feature.text_features import (
    get_text_features
)
from src.preprocessing.preprocess import preprocess


#%%

reload_all()
data = read_pdfs()

#%%
preprocess(data)

#%%


#%%
from src.feature.text_features import (_get_topics)
_get_topics(data.documents)



#%%
import os

current_path = os.getcwd()
print(current_path)

# from pathlib import path
# print()
y = []
with open('data/labels/types.txt') as f:
    text = f.read()
    y = text.split('\n')



from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch

labels=['email', 'report', 'image','interview','other']

tokenizer = RobertaTokenizer.from_pretrained("distilroberta-base")
model = RobertaForSequenceClassification.from_pretrained(
    "distilroberta-base",
    num_labels=len(labels)
)

# y_l = [
#     "report","email","other","email","email","email","report",
#     "image","report","report","report","report","report","report",
#     "interview","interview","interview","report","report","report",
#     "other","image","other","report","report","other",
#     "image","other","report","report","image","report","email",
#     "report","report","report","report","report","email","email","email",
#     "report","report","email","report","report","report","report","report",
#     "report"
# ]

label2id = {label: i for i, label in enumerate(labels)}
y = [label2id[lbl] for lbl in y]

docs = data.documents
texts = [doc.get_full_content() for doc in docs]

encodings = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

outputs = model(**encodings, labels=y)
loss = outputs.loss
logits = outputs.logits
predictions = torch.argmax(logits, dim=1)

#%%
from sklearn.feature_extraction.text import CountVectorizer

def infer_type(dataset: Dataset):
    model = LogisticRegression()
    docs = data.documents
    texts = [" ".join(doc.get_token()) for doc in docs]

    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)

    predictions = model.fit_transform(X,y)
    print(predictions)


#%% LOAD DATA

reload_all()
data = read_pdfs()
preprocess(data)
get_text_features(data)