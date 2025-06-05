from src.run_training import train_and_evaluate


def test_training_accuracy():
    acc = train_and_evaluate()
    assert 0.9 <= acc <= 1.0
