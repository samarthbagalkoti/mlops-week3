import json
from pathlib import Path

ARTIFACTS = Path("artifacts")

def test_metrics_exist():
    """Ensure metrics.json file is created after evaluation."""
    metrics_file = ARTIFACTS / "metrics.json"
    assert metrics_file.exists(), "metrics.json missing!"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    assert "accuracy" in metrics, "Accuracy not recorded!"
    assert "f1_macro" in metrics, "F1 not recorded!"

def test_metrics_threshold():
    """Check that accuracy is above a reasonable threshold."""
    metrics_file = ARTIFACTS / "metrics.json"
    with open(metrics_file, "r") as f:
        metrics = json.load(f)
    acc = metrics["accuracy"]
    assert acc > 0.7, f"Accuracy too low! Got {acc}"

