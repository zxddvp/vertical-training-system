import pandas as pd
from src.data_processing import load_dataset, vertical_split
from src.train import train
from src.evaluate import accuracy


def test_training():
    df = load_dataset('data/financial.csv')
    parts = vertical_split(df, [['age', 'income'], ['credit_score']], 'label')
    model, merged = train(parts, 'label', epochs=1)
    acc = accuracy(model, merged, 'label')
    assert 0.0 <= acc <= 1.0

if __name__ == '__main__':
    test_training()