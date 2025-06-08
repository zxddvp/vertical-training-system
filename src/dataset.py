from typing import List, Tuple
import random


def load_dummy_dataset() -> Tuple[List[List[float]], List[int]]:
    """Return a small linearly separable dataset."""
    X = [[0.0, 0.0], [0.1, 0.2], [0.8, 0.7], [1.0, 0.9], [0.2, 0.1], [0.9, 1.0]]
    y = [0, 0, 1, 1, 0, 1]
    return X, y


def train_test_split(
    X: List[List[float]],
    y: List[int],
    test_ratio: float = 0.25,
    seed: int | None = None,
) -> Tuple[Tuple[List[List[float]], List[int]], Tuple[List[List[float]], List[int]]]:
    """Split data into train and test sets."""
    indices = list(range(len(X)))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return (X_train, y_train), (X_test, y_test)
