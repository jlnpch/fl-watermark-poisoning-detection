# quickstart-pytorch — Agent Guide

## Project

Flower federated learning example using PyTorch + CIFAR-10.  
ServerApp uses FedAvg; ClientApp runs train/evaluate on partitioned CIFAR-10.

## Key files

- `pytorchexample/task.py` — model (`Net` CNN), data loading, train/test
- `pytorchexample/server_app.py` — `ServerApp` with `FedAvg` strategy
- `pytorchexample/client_app.py` — `ClientApp` with `@app.train()` and `@app.evaluate()`
- `pyproject.toml` — dependencies, Flower config, default run config

## Commands

```sh
pip install -e .              # install project + deps
flwr run . --stream           # run simulation (default config)
flwr run . --run-config "num-server-rounds=5 learning-rate=0.05" --stream
```

Default config overridable: `num-server-rounds`, `fraction-evaluate`, `local-epochs`, `learning-rate`, `batch-size`, `save-model`.

## Structure

- Single package `pytorchexample/`, no monorepo
- Entrypoints: `serverapp = pytorchexample.server_app:app`, `clientapp = pytorchexample.client_app:app`
- No tests, no lint/typecheck config
