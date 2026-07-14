# quickstart-pytorch — Agent Guide

## Project location
Run from **local NVMe**: `/home2/julian/test/quickstart-pytorch/`  
Original (reference copy) at `/mnt/MSF-NAS/home/julian/test/quickstart-pytorch/`.  
Dataset cache symlinked to `/home2/julian/.cache/huggingface/datasets/`.

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

Uchida-style watermark embedded **before** FL starts via server-side pretraining on a fraction of CIFAR-10 training data with combined loss: `L = L_task + λ·||P·W − b||²`.  

Each round, per-client `watermark_ber` is logged server-side to track watermark degradation.  
No regularization applied during client training — pure observation.

Config (all overridable via `--run-config`):
- `watermark-message` (default `"uchida"`) — seed string for watermark bits
- `watermark-num-bits` (default `64`) — watermark length
- `watermark-lambda` (default `0.01`) — watermark regularization weight during server pretraining
- `pretrain-fraction` (default `0.1`) — fraction of CIFAR-10 training data used by server for pretraining
- `pretrain-epochs` (default `5`) — number of server pretraining epochs

## Commands

```sh
# Run from local NVMe for speed:
cd /home2/julian/test/quickstart-pytorch
PATH="/home/julian/.local/bin:$PATH" flwr run . --stream           # default config
PATH="/home/julian/.local/bin:$PATH" flwr run . --run-config "num-server-rounds=5 learning-rate=0.01" --stream
```

Default config overridable: `num-server-rounds`, `fraction-evaluate`, `local-epochs`, `learning-rate`, `batch-size`, `save-model`, `watermark-message`, `watermark-num-bits`.

## Structure

- Single package `pytorchexample/`, no monorepo
- Entrypoints: `serverapp = pytorchexample.server_app:app`, `clientapp = pytorchexample.client_app:app`
- Dataset: CIFAR-10, IID partition, 80/20 train/test split per client
- No tests, no lint/typecheck config
