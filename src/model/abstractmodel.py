import abc
from pathlib import Path
from typing import Any

from numpy.typing import ArrayLike


class AbstractModel(abc.ABC):
    @abc.abstractmethod
    def fit(
        self,
        X: ArrayLike,
        *,
        X_val: ArrayLike | None = None,
    ) -> Any:
        """Train the model on the provided data."""

    @abc.abstractmethod
    def predict(self, X: ArrayLike) -> ArrayLike:
        """Make predictions based on the trained model."""

    @abc.abstractmethod
    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        """Predict class probabilities."""

    @abc.abstractmethod
    def reset_model(self) -> None:
        """Reset the model."""

    @abc.abstractmethod
    def save_model(self, path: Path) -> None:
        """Save the model to path."""

    @abc.abstractmethod
    def load_model(self, path: Path) -> "AbstractModel":
        """Load the model from path."""