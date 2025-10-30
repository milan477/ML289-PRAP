# %%
from src.io.read import read_pdfs
from src.schema.dataset import Dataset


#%% LOAD DATA

data = read_pdfs()
print(data)


#%%
print(type(data))
print(len(data),"files")