import os

from src.train import train


def test_model_accuracy_and_artifacts(tmp_path):
    accuracy = train(model_dir=str(tmp_path))
    assert accuracy >= 0.9
    assert os.path.exists(tmp_path / "model.joblib")
    assert os.path.exists(tmp_path / "metrics.json")
