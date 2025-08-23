# tests/test_training.py
import hashlib
import json
from pathlib import Path
import subprocess, shutil

ARTIFACTS = Path("artifacts")

def file_hash(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def test_model_artifacts_exist():
    """Check if training artifacts exist after make train."""
    assert (ARTIFACTS / "model.joblib").exists(), "Model file missing!"
    assert (ARTIFACTS / "split.npz").exists(), "Data split file missing!"
    assert (ARTIFACTS / "params_used.json").exists(), "Params file missing!"

def test_deterministic_training():
    """
    Train twice with same seed and compare model binaries by hash.
    IMPORTANT: Copy models OUTSIDE artifacts before deleting artifacts.
    """
    tmp1 = Path("model1.joblib")  # outside artifacts
    tmp2 = Path("model2.joblib")  # outside artifacts

    # 1st run (fresh)
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir()
    subprocess.run(["python", "src/train.py", "--config", "configs/params.json"], check=True)
    shutil.copy2(ARTIFACTS / "model.joblib", tmp1)

    # 2nd run (fresh again)
    shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir()
    subprocess.run(["python", "src/train.py", "--config", "configs/params.json"], check=True)
    shutil.copy2(ARTIFACTS / "model.joblib", tmp2)

    # Compare strong hashes (better than just file size)
    h1, h2 = file_hash(tmp1), file_hash(tmp2)
    assert h1 == h2, f"Model binaries differ! {h1} vs {h2}"

    # Cleanup temp files
    tmp1.unlink(missing_ok=True)
    tmp2.unlink(missing_ok=True)

