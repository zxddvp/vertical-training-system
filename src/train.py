import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import List

from .data_processing import combine_vertical

from .models import SimpleClassifier


def dataframe_to_tensor(df, label_col: str) -> TensorDataset:
    labels = torch.tensor(df[label_col].values, dtype=torch.long)
    features = torch.tensor(df.drop(columns=[label_col]).values, dtype=torch.float32)
    return TensorDataset(features, labels)


def train(dataset_parts: List, label_col: str, epochs: int = 5):
    datasets = [dataframe_to_tensor(df, label_col) for df in dataset_parts]
    loaders = [DataLoader(ds, batch_size=4, shuffle=True) for ds in datasets]
    input_dim = datasets[0].tensors[0].shape[1]
    model = SimpleClassifier(input_dim)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        for loader in loaders:
            for x, y in loader:
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
        print(f"epoch {epoch+1} loss: {loss.item():.4f}")

    merged = combine_vertical(dataset_parts, label_col)
    return model, merged


def hybrid_parallel_train(*args, **kwargs):
    """混合并行训练占位函数。"""
    pass


def distillation_train(*args, **kwargs):
    """分布式知识蒸馏占位函数。"""
    pass
