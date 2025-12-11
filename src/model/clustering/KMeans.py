from src.model.abstractmodel import AbstractModel
from numpy.typing import ArrayLike

from sklearn.cluster import KMeans
from typing import Any
from typing_extensions import override


class KMeansModel(AbstractModel):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs
        self.reset_model()

    @override
    def fit(self, X: ArrayLike,  *, X_val: ArrayLike | None = None) -> Any:
        self.model.fit(X)
        print(f"KMeans converged: {self.model.n_iter_} iterations")

    @override
    def predict(self, X: ArrayLike) -> ArrayLike:
        return None

    @override
    def predict_proba(self, X):
        return None

    @override
    def reset_model(self):
        self.model = KMeans(*self.args, **self.kwargs)

    @override
    def save_model(self, path):
        return None

    @override
    def load_model(self, path):
        return None
