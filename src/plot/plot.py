import seaborn as sns
import matplotlib.pyplot as plt

def plot_histogram(values, bins:int=8, ax=None, show = True, title = "", xlabel = "", ylabel = ""):
    created = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        created = True

    sns.histplot(values, bins=bins, kde=False, discrete=True, ax=ax, palette="viridis")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if created:
        plt.tight_layout()
        plt.show()


def plot_countplot(values, ax=None, order=None, title="",xlabel="", ylabel=""):
    created = False
    if not ax:
        fig, ax = plt.subplots(figsize=(10, 5))
        created = True

    sns.countplot(x=values, ax=ax, palette="viridis", order=order)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if created:
        plt.tight_layout()
        plt.show()