from typing import Tuple, List

from src.dataset import load_dummy_dataset, train_test_split
from src.model import LogisticRegression
from src.evaluate import compute_accuracy


def train_and_evaluate() -> float:
    X, y = load_dummy_dataset()
    (X_train, y_train), (X_test, y_test) = train_test_split(X, y, seed=42)
    model = LogisticRegression(n_features=len(X_train[0]))
    model.fit(X_train, y_train, epochs=200, lr=0.5)
    accuracy = compute_accuracy(model, (X_test, y_test))
    return accuracy


if __name__ == "__main__":
    acc = train_and_evaluate()
    print(f"Test Accuracy: {acc * 100:.2f}%")
