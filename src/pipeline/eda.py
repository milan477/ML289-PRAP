# %%
import importlib

import src.io.read as read
importlib.reload(read)

import src.schema.dataset as dataset
importlib.reload(dataset)

import src.feature.text_features as text_features
importlib.reload(text_features)

import src.feature.image_features as image_features
importlib.reload(image_features)



from src.io.read import read_pdfs

from src.schema.dataset import Dataset

from src.feature.image_features import (
get_image_features
)

from src.feature.text_features import (
get_text_features
)



#%% LOAD DATA

data = read_pdfs()


#%%
# Feature 1
get_image_features(data)
get_text_features(data)