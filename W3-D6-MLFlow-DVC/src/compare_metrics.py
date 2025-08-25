import json, argparse, sys
from pathlib import Path

def read_json(p: Path):
    with open(p, "r") as f:
        return json.load(f)

def main(current, baseline, metric, max_drop):
    current_m = read_json(Path(current))
    baseline_m = read_json(Path(baseline))
    if metric not in current_m or metric not in baseline_m:
        print(f"[gate] Metric '{metric}' not found in one of the files.")
        sys.exit(2)

    curr = float(current_m[metric])
    base = float(baseline_m[metric])
    drop = max(0.0, base - curr)  # positive drop means worse
    allowed = float(max_drop)

    print(f"[gate] {metric}: baseline={base:.4f}, current={curr:.4f}, drop={drop:.4f}, allowed={allowed:.4f}")

    if drop > allowed:
        print("[gate] ❌ Regression detected. Failing gate.")
        sys.exit(2)
    else:
        print("[gate] ✅ Within threshold. Gate passed.")
        sys.exit(0)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", required=True, help="artifacts/metrics.json")
    ap.add_argument("--baseline", required=True, help="artifacts/metrics_baseline.json")
    ap.add_argument("--metric", default="accuracy")
    ap.add_argument("--max-drop", type=float, default=0.05)
    args = ap.parse_args()
    main(args.current, args.baseline, args.metric, args.max_drop)

