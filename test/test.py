import src.io.read as read




from src.plot.plot import plot_histogram
from src.preprocessing.preprocess import preprocess

import src.schema.dataset as ds
import importlib
importlib.reload(ds)
importlib.reload(read)

from src.io.read import read_pdfs

data = read_pdfs()
# preprocess(data)


#%%
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

lengths = []
for document in data.documents:
    lengths.append(len(document))
print(lengths)
import pandas as pd
s=pd.Series(lengths)
s=s[s<130]

sns.boxplot(x=s.values,)
plt.xlabel("Number of Pages")
plt.ylabel("Number of Documents")
plt.title("Document Page Count")
plt.show()

#%%
print(len(lengths))