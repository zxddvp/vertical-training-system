from typing import Iterable, Tuple, List, Protocol

class ModelProtocol(Protocol):
    def predict(self, x: Iterable[float]) -> int:
        ...

def compute_accuracy(model: ModelProtocol, dataset: Tuple[List[List[float]], List[int]]) -> float:
    """Compute classification accuracy for the given model and dataset."""
    X, y = dataset
    correct = 0
    for xi, yi in zip(X, y):
        pred = model.predict(xi)
        if pred == yi:
            correct += 1
    return correct / len(y)
