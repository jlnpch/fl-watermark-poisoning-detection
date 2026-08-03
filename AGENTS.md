# quickstart-pytorch — Agent Guide

## Project location
Run from **local NVMe**: `/home2/julian/test/quickstart-pytorch/`  
Dataset cache on local disk: `/home2/julian/.cache/huggingface/datasets/`

Flower federated learning example using PyTorch + CIFAR-10.
ServerApp uses `WatermarkedFedAvg` (extends `FedAvg`); ClientApp runs train/evaluate on IID-partitioned CIFAR-10.

## Key files

- `pytorchexample/task.py` — model (`Net` CNN), data loading, train/test
- `pytorchexample/server_app.py` — `ServerApp` with `WatermarkedFedAvg` strategy
- `pytorchexample/client_app.py` — `ClientApp` with `@app.train()` and `@app.evaluate()`
- `pytorchexample/watermark.py` — `UchidaWatermark` class (embed + BER computation on `fc3.weight`)
- `pytorchexample/watermarked_strategy.py` — `WatermarkedFedAvg` (logs per-client watermark BER each round)
- `pytorchexample/metrics.py` — CSV saver (train CSV has `watermark_ber` + `excluded` columns)
- `pytorchexample/run_registry.py` — JSONL registry of all runs with configs
- `pyproject.toml` — dependencies, Flower config, default run config (results-dir is **absolute**: `/home2/julian/test/quickstart-pytorch/results`)

## CRITICAL: Running Experiments

### Step-by-step

```sh
# 1. Kill stale processes from previous runs (GPU cleanup)
kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 5
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
sleep 3

# 2. Start superlink
PATH="/home/julian/.local/bin:$PATH" flower-superlink --insecure --simulation &
sleep 5

# 3. Configure federation (10 SuperNodes + GPU per client). This is MANDATORY —
#    Flower 1.32 defaults to only 2 SuperNodes, and GPU is disabled without these flags.
PATH="/home/julian/.local/bin:$PATH" flwr federation simulation-config @none/default local \
    --num-supernodes 10 \
    --client-resources-num-gpus 0.1 \
    --init-args-num-gpus 1

# 4. Run the experiment
PATH="/home/julian/.local/bin:$PATH" flwr run /home2/julian/test/quickstart-pytorch --stream \
    --run-config 'key1=val1 key2=val2'
```

### Run Config Quoting Rules (IMPORTANT — causes cryptic "invalid format" errors)

The `--run-config` value is a space-separated string of `key=value` pairs. The parser
uses strict type inference based on the value format. Follow these rules:

| Value type | Example | Quote rule |
|---|---|---|
| Integer | `num-server-rounds=50` | no quotes |
| Float | `watermark-lambda=0.01` | no quotes |
| Boolean | `save-model=false` | no quotes (lowercase) |
| String (no special chars) | `partition-type=iid` | no quotes |
| **String with underscores** | `attacker-type="sign_flip"` | **double quotes required** |
| **Python keyword `none`** | `attacker-type="none"` | **double quotes required** (bare `none` is parsed as Python `None`) |

**Correct outer quoting for shell** (use single quotes around the whole run-config to avoid
nested quoting issues):

```sh
# Good — single-quote outer, escaped inner quotes:
--run-config 'attacker-type="sign_flip" watermark-lambda=0.01 max-trusted-ber=0.25'

# Also good — double-quote outer, escaped inner quotes:
--run-config "attacker-type=\"sign_flip\" watermark-lambda=0.01"

# BAD — bare none gets parsed as Python None:
--run-config 'attacker-type=none'          # ERROR
--run-config 'attacker-type="none"'        # OK
```

### Full run config keys (from pyproject.toml + server code)

| Key | Default | Type | Notes |
|---|---|---|---|
| `attacker-type` | `none` | string | `none`, `noise`, `sign_flip`, `label_flip` — must be quoted in run-config |
| `attacker-fraction` | `0.1` | float | Fraction of clients that are malicious |
| `attacker-noise-scale` | `1.0` | float | Noise attack: std of Gaussian noise added to update |
| `sign-flip-scale` | `1.0` | float | Sign-flip: multiplier applied after sign inversion |
| `label-flip-source` | `9` | int | Label-flip: source class to flip from |
| `label-flip-target` | `2` | int | Label-flip: target class to flip to |
| `label-flip-scale` | `1.0` | float | Label-flip: amplification of the update residual |
| `max-trusted-ber` | `0.30` | float | Defense threshold: clients with BER above this are excluded |
| `watermark-lambda` | `0.01` | float | Watermark regularization weight during pretraining |
| `num-server-rounds` | `50` | int | Number of FL rounds |
| `pretrain-epochs` | `10` | int | Server pretrain epochs |
| `server-private-samples` | `5000` | int | Samples reserved for server pretraining |
| `client-samples` | `4000` | int | Training samples per client |
| `early-stopping-patience` | `0` | int | 0 = disabled |
| `learning-rate` | `0.01` | float | SGD learning rate |
| `weight-decay` | `0.0005` | float | SGD weight decay |
| `batch-size` | `64` | int | Train/eval batch size |
| `local-epochs` | `1` | int | Client local training epochs |
| `fraction-evaluate` | `1.0` | float | Fraction of clients used for evaluation |

### Data flow

- Server takes first `server-private-samples` from shuffled CIFAR-10 for pretraining.
- Clients receive `client-samples` each from the remaining pool (capped at 50k - server_private_samples).

## GPU Cleanup

When runs crash or are aborted, stale `flwr-simulation` and `ray` processes leak GPU memory.
To restore clean state (1 MiB baseline):

```sh
kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
sleep 3
nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
sleep 2
```

## Output

Results save to `/home2/julian/test/quickstart-pytorch/results/` (absolute path set in `pyproject.toml`):
- `run_{timestamp}_server.csv` — round, server_acc, server_loss, server_asr
- `run_{timestamp}_train.csv` — round, partition_id, train_loss, watermark_ber, is_attacker, excluded
- `run_{timestamp}_eval.csv` — round, partition_id, eval_acc, eval_loss
- `run_registry.jsonl` — one JSON line per run with all config keys

Every run generates a unique run_id (timestamp). The train CSV is the source for BER and exclusion analysis.

### Plot Directory Structure

```
results/plots/
├── baseline/           # No-attack baseline sweep (pretrain-epochs × dataset size)
│   ├── baseline_ber.png           # BER evolution (4 server sizes × 3 pretrain epochs)
│   └── baseline_acc.png           # Accuracy evolution
├── 5000_30ep/          # Config: 5000 samples, 30 pretrain epochs
│   ├── baseline_lambda_scan.png   # Clean baseline BER/acc vs λ (N=10 per λ)
│   ├── baseline_lambda_ber_evolution.png  # BER over rounds, all λ overlaid
│   ├── eval_ber_evolution.png     # Honest vs attacker BER per config
│   ├── eval_accuracy.png          # Server accuracy per config
│   ├── eval_defense_summary.png   # d', detection rate, defense accuracy
│   └── results_tables.txt
└── 5000_50ep/          # Config: 5000 samples, 50 pretrain epochs
    ├── baseline_lambda_scan.png   # N=5 per λ
    ├── baseline_lambda_ber_evolution.png
    ├── eval_ber_evolution.png
    ├── eval_accuracy.png
    ├── eval_defense_summary.png
    └── results_tables.txt
```

Plot scripts take `--pretrain-epochs N` and write to `results/plots/5000_{N}ep/`.
The `baseline/` sweep curves are pinned via `SWEEP_EXCLUDE_PREFIXES`
(`("20260730", "20260731")`) so later lambda-study batches don't change them.

## Sweep Script Template

```sh
#!/bin/bash
set -e
VALUES=(0.005 0.01 0.05 0.1)

for VAL in "${VALUES[@]}"; do
    echo "Starting run with val=${VAL}"

    # Cleanup
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 5
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    sleep 3

    # Start superlink
    PATH="/home/julian/.local/bin:$PATH" flower-superlink --insecure --simulation &
    sleep 5

    # Configure federation
    PATH="/home/julian/.local/bin:$PATH" flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1

    # Run
    PATH="/home/julian/.local/bin:$PATH" flwr run /home2/julian/test/quickstart-pytorch --stream \
        --run-config "key1=val1 key2=\"string_val\" key3=${VAL}" \
        > "/tmp/run_${VAL}.log" 2>&1

    echo "Run ${VAL} finished"
done

echo "ALL DONE"
```

Notes:
- Use `|| true` after kill commands to prevent `set -e` from aborting on no matching processes.
- Use `setsid` to launch the sweep in background: `setsid bash script.sh > log 2>&1 &`
- Each run takes ~10 min with GPU (30 pretrain + 50 FL rounds).
- The path `/home/julian/.local/bin/flwr` may be used when `flwr` is not in the default PATH.

## Known Issues & Pitfalls

1. **GPU disabled by default**: Without `--client-resources-num-gpus 0.1 --init-args-num-gpus 1` in
   `simulation-config`, Ray sets `CUDA_VISIBLE_DEVICES=""` and clients train on CPU (10× slower).
2. **`set -e` + kill**: `kill` with empty argument list (no matching processes) returns non-zero.
   Always suffix with `|| true`.
3. **`none` string**: The config parser converts bare `none` to Python `None`. Always write `"none"`.
4. **Underscore strings**: `sign_flip`, `label_flip` need double quotes.
5. **Hyphenated keys**: All config keys use hyphens (e.g. `watermark-lambda`), not underscores.
   This is Flower convention.
6. **No model saving during sweep**: `save-model=false` by default. Set to `true` to dump
   `final_model.pt` (only the last run's model survives, in the CWD of the runtime env).
7. **Run registry**: `run_registry.jsonl` is appended to, never overwritten. If the file doesn't
   exist at the start of a run, it is created from scratch. Old entries are preserved across runs
   as long as the file persists.
8. **Registry duplicates**: some run_ids appear multiple times in `run_registry.jsonl` (with
   different `watermark-lambda` values). Dedup by `run_id` before counting runs; the train CSV
   filename is derived from the run_id and is the source of truth.

## Baseline Sweep

`utility scripts/run_sweep.sh` runs a baseline hyperparameter sweep:
- Server private samples: 500, 2500, 5000, 10000
- Pretrain epochs: 10, 30, 50
- 3 repetitions per pair (37 total runs)
- 50 FL rounds, IID, no attacks, 10 supernodes

`utility scripts/plot_baseline.py` generates BER and accuracy evolution plots from sweep results.
Output: `results/plots/baseline/baseline_ber.png` and `baseline_acc.png` (plus the λ scan and
BER-evolution plots; the latter two include the N=10 lambda consistency study).

## Semi-Fragility Evaluation

`utility scripts/run_evaluation.sh` evaluates watermark detection under different attacks
(for 5000/30 config):
- Label flip (scale=5, λ=0.01)
- Sign flip (scale=5, λ=0.005/0.01/0.05/0.10/0.25/0.75)
- 3 repetitions per config (7 configs, 21 runs)
- 50 FL rounds, IID, 10 supernodes, BER threshold 0.25

`utility scripts/plot_evaluation.py` generates defense analysis plots.
Output: `results/plots/5000_30ep/eval_*.png`

## 5000/50 config

- `utility scripts/run_lambda_sweep_50ep.sh` — clean baseline lambda sweep (6 λ × 5 reps = 30 runs)
- `utility scripts/run_defense_50ep.sh` — defense evaluation (1 label_flip + 6 sign_flip × 3 reps = 21 runs)
- Plots: `results/plots/5000_50ep/*.png`
- Results (N=5 sweep, N=3 defense): no BER–λ trend; defense accuracy uniformly higher
  than 30ep (94.0–99.2% vs 89.8–97.8%).
