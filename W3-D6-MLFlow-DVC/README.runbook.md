# W3:D7 — Reproducible Training + Evaluation Pipeline

## Problem
Build a reproducible ML training pipeline with evaluation, testing, CI/CD, and experiment tracking.

## Architecture
- Config-driven training (`config.yaml`)
- Metrics + gates (`metrics.json`, regression threshold 5%)
- Tests (`pytest`)
- CI/CD (`GitHub Actions` workflow)
- Experiment tracking (`MLflow`)
- Artifact versioning (`DVC`)

## Deliverables
- Deterministic pipeline (seeds, params logged)
- Quality gates (fail if accuracy drop >5%)
- Automated tests (pytest)
- CI/CD badge in repo
- MLflow UI logs + DVC-tracked artifacts

## How to Run
```bash
make setup
make run
make baseline
make gate
make test
mlflow ui

