import sys
import json
import shutil
import subprocess
from pathlib import Path
from joblib import load

ARTIFACTS = Path("artifacts")
CONFIG = "configs/config.yaml"   # ← we moved to YAML in W3:D4

def run_py(args):
    """Run using the current Python interpreter (venv-safe)."""
    return subprocess.run([sys.executable, *args], check=True)

def test_model_artifacts_exist():
    # Ensure a clean train/eval first
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir()

    run_py(["src/train.py", "--config", CONFIG])
    assert (ARTIFACTS / "model.joblib").exists(), "Model file missing!"
    assert (ARTIFACTS / "split.npz").exists(), "Data split file missing!"
    assert (ARTIFACTS / "params_used.json").exists(), "Params file missing!"

def test_deterministic_training():
    """
    Train twice with same seed and compare model binaries by size/hash proxy.
    IMPORTANT: Move models OUTSIDE artifacts before wiping it.
    """
    tmp1 = Path("model1.joblib")
    tmp2 = Path("model2.joblib")

    # 1st run
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir()
    run_py(["src/train.py", "--config", CONFIG])
    (ARTIFACTS / "model.joblib").replace(tmp1)

    # 2nd run
    shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir()
    run_py(["src/train.py", "--config", CONFIG])
    (ARTIFACTS / "model.joblib").replace(tmp2)

    size1, size2 = tmp1.stat().st_size, tmp2.stat().st_size
    assert size1 == size2, f"Models differ: {size1} vs {size2}"

def test_metrics_exist_and_threshold():
    # Evaluate once to generate metrics
    if not (ARTIFACTS / "metrics.json").exists():
        run_py(["src/evaluate.py", "--config", CONFIG])

    metrics_path = ARTIFACTS / "metrics.json"
    assert metrics_path.exists(), "metrics.json missing!"
    metrics = json.loads(metrics_path.read_text())
    assert "accuracy" in metrics, "Accuracy not recorded!"
    assert "f1_macro" in metrics, "F1 not recorded!"
    assert metrics["accuracy"] > 0.7, f"Accuracy too low! {metrics['accuracy']}"

