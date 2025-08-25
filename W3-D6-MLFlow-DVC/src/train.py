import mlflow
import yaml
import os, json, argparse, random, hashlib
from pathlib import Path

import numpy as np
from joblib import dump
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

ARTIFACTS = Path("artifacts")
SPLIT_FILE = ARTIFACTS / "split.npz"
MODEL_FILE = ARTIFACTS / "model.joblib"
PARAMS_USED_FILE = ARTIFACTS / "params_used.json"
DATA_HASH_FILE = ARTIFACTS / "data_hash.txt"

def set_seeds(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(cfg_path: str):
    ...
    # Start MLflow run
    with mlflow.start_run():
        mlflow.log_params(cfg)     # log all params
        mlflow.log_param("model", model_name)

        # Train model
        model.fit(X_train, y_train)

        # Save model
        dump(model, MODEL_FILE)
        mlflow.log_artifact(str(MODEL_FILE), artifact_path="models")

        # Save params
        with open(PARAMS_USED_FILE, "w") as f:
            json.dump({"seed": seed, "model": model_name, model_name: cfg.get(model_name, {})}, f, indent=2)
        mlflow.log_artifact(str(PARAMS_USED_FILE), artifact_path="params")


def load_config(path: str) -> dict:
    # Accept both YAML and JSON based on file extension
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")):
            return yaml.safe_load(f)
        elif path.endswith(".json"):
            return json.load(f)
        else:
            # Fallback: try YAML, then JSON
            try:
                return yaml.safe_load(f)
            except Exception:
                f.seek(0)
                return json.load(f)

def hash_array(arr: np.ndarray) -> str:
    m = hashlib.sha256()
    m.update(arr.tobytes())
    return m.hexdigest()

def main(cfg_path: str):
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    cfg = load_config(cfg_path)
    seed = int(cfg.get("seed", 42))
    set_seeds(seed)

    # Load dataset (Iris, tiny and deterministic)
    iris = load_iris()
    X = iris["data"]
    y = iris["target"]

    # Save a data "fingerprint" (helps prove data didn't change)
    with open(DATA_HASH_FILE, "w") as f:
        f.write(f"shape={X.shape}, sha256={hash_array(X)}\n")

    # Create or reuse split (so eval uses same indices)
    if SPLIT_FILE.exists():
        split = np.load(SPLIT_FILE)
        train_idx, test_idx = split["train_idx"], split["test_idx"]
    else:
        idx = np.arange(len(y))
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X, y, idx, test_size=cfg["test_size"], random_state=seed, stratify=y
        )
        # Save only indices; eval will re-materialize X/y
        np.savez(SPLIT_FILE, train_idx=train_idx, test_idx=test_idx)

    # Materialize split
    X_train, y_train = X[train_idx], y[train_idx]

    # Build model
    model_name = cfg.get("model", "logreg")
    if model_name == "logreg":
        p = cfg.get("logreg", {})
        model = LogisticRegression(
            C=p.get("C", 1.0),
            max_iter=p.get("max_iter", 1000),
            solver=p.get("solver", "liblinear"),
            random_state=seed
        )
    elif model_name == "rf":
        p = cfg.get("rf", {})
        model = RandomForestClassifier(
            n_estimators=p.get("n_estimators", 100),
            max_depth=p.get("max_depth", None),
            n_jobs=p.get("n_jobs", -1),
            random_state=seed
        )
    else:
        raise ValueError(f"Unknown model '{model_name}'")

    # Train
    model.fit(X_train, y_train)

    # Save artifacts
    dump(model, MODEL_FILE)
    with open(PARAMS_USED_FILE, "w") as f:
        json.dump({"seed": seed, "model": model_name, model_name: cfg.get(model_name, {})}, f, indent=2)

    print(f"[train] Saved model -> {MODEL_FILE}")
    print(f"[train] Saved split  -> {SPLIT_FILE}")
    print(f"[train] Saved params -> {PARAMS_USED_FILE}")
    print(f"[train] Data hash    -> {DATA_HASH_FILE}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to config YAML/JSON")
    args = ap.parse_args()
    main(args.config)

