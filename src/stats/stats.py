from src.schema.dataset import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from src.plot.plot import plot_histogram, plot_countplot
import pandas as pd

def get_nr_of_documents(dataset: Dataset):
    dataset_stats = {}
    dataset_stats['length'] = len(dataset)
    return dataset_stats

def get_documents_page_count_histogram(dataset: Dataset, bins: int = 8):
    lengths=[]
    for document in dataset.documents:
        lengths.append(len(document))
    nr_of_bins = min(bins, max(lengths))
    return np.histogram(lengths, bins=nr_of_bins, range =(0,max(lengths)))

def plot_documents_page_count(dataset: Dataset, ax=None):
    lengths = []
    for document in dataset.documents:
        lengths.append(len(document))
        print(type(len(document)))
    plot_countplot(values=lengths, order=[1,2,3,4,5], title="Document Page Count", xlabel="Number of Pages", ylabel="Number of Documents")