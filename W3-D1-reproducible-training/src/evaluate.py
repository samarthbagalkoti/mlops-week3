import json, argparse
from pathlib import Path

import numpy as np
from joblib import load
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ARTIFACTS = Path("artifacts")
SPLIT_FILE = ARTIFACTS / "split.npz"
MODEL_FILE = ARTIFACTS / "model.joblib"
METRICS_FILE = ARTIFACTS / "metrics.json"
BASELINE_FILE = ARTIFACTS / "metrics_baseline.json"

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)

def main(cfg_path: str):
    cfg = load_config(cfg_path)
    if not SPLIT_FILE.exists():
        raise FileNotFoundError("Split not found. Run `make train` first.")
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run `make train` first.")

    # Load data and split
    iris = load_iris()
    X, y = iris["data"], iris["target"]
    split = np.load(SPLIT_FILE)
    test_idx = split["test_idx"]
    X_test, y_test = X[test_idx], y[test_idx]

    # Load model
    model = load(MODEL_FILE)

    # Predict & metrics
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_test": int(len(y_test))
    }

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Saved metrics -> {METRICS_FILE}")
    print(json.dumps(metrics, indent=2))

    # First-time convenience: auto-create baseline if not present
    if not BASELINE_FILE.exists():
        with open(BASELINE_FILE, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[eval] Baseline metrics created -> {BASELINE_FILE}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)

