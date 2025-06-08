import torch
from torch.utils.data import DataLoader

from .train import dataframe_to_tensor


def accuracy(model: torch.nn.Module, df, label_col: str) -> float:
    """计算模型在给定数据集上的准确率."""
    model.eval()
    dataset = dataframe_to_tensor(df, label_col)
    loader = DataLoader(dataset, batch_size=4)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total else 0.0

