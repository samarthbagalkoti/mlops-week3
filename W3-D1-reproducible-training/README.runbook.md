# W3-D1 — Reproducible Training + Metric Gate

## Quickstart
```bash
make setup
make run        # train + eval
make baseline   # (Optional) lock today's metrics as baseline
make gate       # compare current vs baseline (fails on regression)

