import yaml
import json, argparse
from pathlib import Path

import numpy as np
from joblib import load
from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt

ARTIFACTS = Path("artifacts")
SPLIT_FILE = ARTIFACTS / "split.npz"
MODEL_FILE = ARTIFACTS / "model.joblib"
METRICS_FILE = ARTIFACTS / "metrics.json"

def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="red")

    fig.tight_layout()
    out_path = ARTIFACTS / "confusion_matrix.png"
    fig.savefig(out_path)
    print(f"[eval] Saved confusion matrix plot -> {out_path}")
    plt.close(fig)

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            return json.load(f)
        else:
            try:
                return yaml.safe_load(f)
            except Exception:
                f.seek(0)
                return json.load(f)

def main(cfg_path: str):
    if not SPLIT_FILE.exists() or not MODEL_FILE.exists():
        raise FileNotFoundError("Run `make train` before evaluate.")

    iris = load_iris()
    X, y = iris["data"], iris["target"]
    split = np.load(SPLIT_FILE)
    test_idx = split["test_idx"]
    X_test, y_test = X[test_idx], y[test_idx]

    model = load(MODEL_FILE)
    y_pred = model.predict(X_test)

    # Basic metrics
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "n_test": int(len(y_test))
    }

    # Save confusion matrix plot
    save_confusion_matrix(y_test, y_pred, labels=iris.target_names)

    # ROC AUC (multi-class handled via "ovr" – one-vs-rest)
    try:
        y_prob = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
        metrics["roc_auc_ovr"] = float(auc)
    except Exception as e:
        print(f"[eval] ROC AUC skipped: {e}")

    with open(METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[eval] Saved metrics -> {METRICS_FILE}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)

