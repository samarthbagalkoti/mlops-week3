PY := .venv/bin/python

.PHONY: setup train eval run baseline gate clean test

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install --upgrade pip
	. .venv/bin/activate && pip install -r requirements.txt

train:
	$(PY) src/train.py --config configs/params.json

eval:
	$(PY) src/evaluate.py --config configs/params.json

run: train eval

# Save today's metrics as baseline (commit this file to lock expectations)
baseline:
	cp artifacts/metrics.json artifacts/metrics_baseline.json

# Fail if accuracy drops >5% vs baseline
gate:
	$(PY) src/compare_metrics.py \
		--current artifacts/metrics.json \
		--baseline artifacts/metrics_baseline.json \
		--metric accuracy --max-drop 0.05

test:
	. .venv/bin/activate && pytest -q

clean:
	rm -rf .venv artifacts/* __pycache__ .pytest_cache
	mkdir -p artifacts

