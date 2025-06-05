import math
from typing import Iterable, List

class LogisticRegression:
    """Simple logistic regression classifier using gradient descent."""

    def __init__(self, n_features: int):
        self.weights: List[float] = [0.0] * n_features
        self.bias: float = 0.0

    def _sigmoid(self, z: float) -> float:
        return 1.0 / (1.0 + math.exp(-z))

    def predict_proba(self, x: Iterable[float]) -> float:
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self._sigmoid(z)

    def predict(self, x: Iterable[float]) -> int:
        return 1 if self.predict_proba(x) >= 0.5 else 0

    def fit(self, X: List[List[float]], y: List[int], epochs: int = 100, lr: float = 0.1) -> None:
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                z = sum(w * xij for w, xij in zip(self.weights, xi)) + self.bias
                p = self._sigmoid(z)
                error = p - yi
                for j in range(len(self.weights)):
                    self.weights[j] -= lr * error * xi[j]
                self.bias -= lr * error
