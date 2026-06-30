# quickstart-pytorch — Agent Guide

## Project

Flower federated learning example using PyTorch + CIFAR-10.  
ServerApp uses `WatermarkedFedAvg` (extends `FedAvg`); ClientApp runs train/evaluate on IID-partitioned CIFAR-10.

## Key files

- `pytorchexample/task.py` — model (`Net` CNN), data loading, train/test
- `pytorchexample/server_app.py` — `ServerApp` with `WatermarkedFedAvg` strategy
- `pytorchexample/client_app.py` — `ClientApp` with `@app.train()` and `@app.evaluate()`
- `pytorchexample/watermark.py` — `UchidaWatermark` class (embed + BER computation on `fc3.weight`)
- `pytorchexample/watermarked_strategy.py` — `WatermarkedFedAvg` (logs per-client watermark BER each round)
- `pyproject.toml` — dependencies, Flower config, default run config

## Watermark

Uchida-style watermark embedded **before** FL starts (via direct projection onto `fc3.weight`).  
Each round, per-client `watermark_ber` is logged server-side to track watermark degradation.  
No regularization applied during client training — pure observation.

Config (all overridable via `--run-config`):
- `watermark-message` (default `"uchida"`) — seed string for watermark bits
- `watermark-num-bits` (default `64`) — watermark length

## Commands

```sh
pip install -e .              # install project + deps
flwr run . --stream           # run simulation (default config)
flwr run . --run-config "num-server-rounds=5 learning-rate=0.01" --stream
```

Default config overridable: `num-server-rounds`, `fraction-evaluate`, `local-epochs`, `learning-rate`, `batch-size`, `save-model`, `watermark-message`, `watermark-num-bits`.

## Structure

- Single package `pytorchexample/`, no monorepo
- Entrypoints: `serverapp = pytorchexample.server_app:app`, `clientapp = pytorchexample.client_app:app`
- Dataset: CIFAR-10, IID partition, 80/20 train/test split per client
- No tests, no lint/typecheck config
