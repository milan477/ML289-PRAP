from src.io.read import read_pdfs

from src.schema.dataset import Dataset

from src.feature.image_features import (
    get_image_features
)

from src.feature.text_features import (
    get_text_features
)

#
data = read_pdfs()
print(get_text_features(data))


# Tokenization using NLTK
# from nltk import word_tokenize, sent_tokenize
# import nltk
#
# for pkg in ["punkt", "punkt_tab", "stopwords"]:
#         try:
#             nltk.data.find(f"tokenizers/{pkg}")
#         except LookupError:
#             nltk.download(pkg, quiet=True)
#
# nltk.download('punkt')
# sent = "GeeksforGeeks is a great learning platform.\
# It is one of the best for Computer Science students."
# print(word_tokenize(sent))
# print(sent_tokenize(sent))

# from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
#
# def remove_stopwords_sklearn(text: str):
#     return [word for word in text.split() if word.lower() not in ENGLISH_STOP_WORDS]
#
# print(remove_stopwords_sklearn("This is an example text."))