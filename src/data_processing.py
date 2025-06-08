import pandas as pd
from typing import List

def load_dataset(path: str) -> pd.DataFrame:
    """加载CSV格式的数据集。"""
    return pd.read_csv(path)

def vertical_split(df: pd.DataFrame, feature_groups: List[List[str]], label_col: str) -> List[pd.DataFrame]:
    """按特征列分组进行垂直切分。返回包含label的子数据集列表。"""
    parts = []
    for group in feature_groups:
        subset_cols = group + [label_col]
        parts.append(df[subset_cols].copy())
    return parts


def combine_vertical(parts: List[pd.DataFrame], label_col: str) -> pd.DataFrame:
    """重新合并垂直切分的数据集。"""
    base = parts[0].copy()
    for part in parts[1:]:
        base = base.join(part.drop(columns=[label_col]))
    return base

