# quickstart-pytorch — Agent Guide

## ▶ CURRENT STATE & RESUME INSTRUCTIONS (updated 2026-11-23)

> Latest commit: `926f569` — "refactor: pluggable defenses + seed control + lock ResNet-18 baseline"

**Correct server path** — the repo lives at **`/home2/julian/test/fl-watermark-poisoning-detection`**
(as used by `pyproject.toml` `results-dir` and every `utility scripts/*.sh`). The
"Project location" line below (quickstart-pytorch) is stale — ignore it.

### Locked baseline config (freeze for all new experiments)
- **Model/dataset:** ResNet-18 (`model-name="resnet18"`), CIFAR-10, 10 classes
- **Watermark:** Uchida 128-bit, `watermark-lambda=0.01`, layer `backbone.fc.weight`
- **Pretraining:** 5000 server samples, `pretrain-epochs=50`, cosine LR (`lr-scheduler="cosine"`)
- **Defense:** dynamic MAD `median + α·MAD`, `mad-alpha=4.0`
- **Seeds:** 3 per run via `seed={0,1,2}` (default 42)
- 50 FL rounds, `batch-size=64`, early stopping disabled

### Pluggable defenses (the infrastructure refactor)
`defense-method` routes aggregation through **`pytorchexample/defenses.py`**
(pure NumPy, no Flower/torch — unit-testable):
`none` | `mad` (default) | `fixed` | `fltrust` | `krum` | `trimmed-mean` | `median`

- `none` = plain FedAvg (fixes the old bug where `none` silently ran MAD).
- `fltrust` trains the server on `fltrust-server-samples` (default 500) root samples each round.
- `krum-m` / `trim-ratio` auto-derive from `attacker-fraction` unless set explicitly.
- `watermarked_strategy.py` now extracts client weights/BER/`num-examples` and calls `run_defense`; it snapshots the global model each round and computes the FLTrust server update via a `server_update_fn` closure from `server_app.py`.

### Seed control
A single `seed` key drives dataset shuffle, Dirichlet partitioning, train/test split, and the watermark projection key (threaded through `task.py`, `watermark.py`, `client_app.py`, `server_app.py`).

### Bugs fixed in this refactor
1. `defense-method="none"` fell through to MAD (CIFAR-100 "baselines" were actually defended).
2. `watermark-layer` was VGG-16's `classifier.6.weight` (crashes ResNet-18) — now `backbone.fc.weight`.
3. `client_app.evaluate()` ignored `partition-type`/`partition-alpha` (non-IID eval loaded IID shards).

### Local unit tests
`PYTHONPATH=. python "utility scripts/test_defenses.py"` — 9 tests, NumPy only.

### Pillar status (2026-11-25)
- ✅ **Pillar A** — done (6 defenses × 4 attacks × 3 seeds, ResNet-18).
- ✅ **Pillar B** — done (freeze-FC evasion → multi-layer fix → escalation → K-scaling → SOTA complementarity).
- ✅ **Pillar C** — done (BadNets stealthy + model-replacement + constrain-and-scale).
- ✅ **Pillar D** — done (non-IID Dirichlet α sweep + attacker-fraction breakdown).

### New config keys (since `926f569`)
`frozen-layers`, `watermark-layers`, `badnets-target`, `badnets-trigger-size`,
`badnets-trigger-value`, `badnets-trigger-pos`, `badnets-poison-frac`, `badnets-scale`,
`badnets-constrain`.
New attack types: `freeze_fc` (freeze watermarked layer, sign-flip rest) and `badnets`
(trigger backdoor; constrain-and-scale via `badnets-constrain`). Multi-layer Uchida
watermark (aggregate BER = max across layers; per-layer BERs logged to `train.csv` via a
JSON `ber_by_layer` column).

---

## Pillar A — ResNet-18 vs Byzantine-robust SOTA (final acc round 50, 3 seeds)

| defense | clean | sign_flip s2 | sign_flip s5 | noise σ=1 |
|---|---|---|---|---|
| none | 0.697 ± 0.003 | 0.410 ± 0.002 | 0.325 ± 0.018 | 0.133 ± 0.017 |
| mad | 0.702 ± 0.011 | 0.645 ± 0.043 | 0.676 ± 0.024 | 0.695 ± 0.009 |
| fltrust | 0.681 ± 0.010 | 0.675 ± 0.002 | 0.681 ± 0.003 | 0.675 ± 0.011 |
| krum | 0.688 ± 0.002 | 0.685 ± 0.005 | 0.680 ± 0.004 | 0.682 ± 0.009 |
| trimmed-mean | 0.699 ± 0.003 | 0.667 ± 0.006 | 0.661 ± 0.002 | 0.695 ± 0.006 |
| median | 0.703 ± 0.004 | 0.667 ± 0.009 | 0.670 ± 0.013 | 0.696 ± 0.005 |

Detection (excluded vs `is_attacker`): `mad` TPR/FPR — s2 0.807/0.018, s5 0.947/0.010,
noise 1.000/0.019. `krum` TPR 1.000 / FPR 0.222 (structural). Clean-run FPR: mad 0.115,
krum 0.200, others 0. Federated wall-clock (s): none/mad ~467, krum ~524, trimmed-mean ~526,
fltrust ~545, median ~590.

## Pillar B — Kerckhoffs-aware attacker & layer-locality

**Part 1 (single-layer FC watermark + MAD, α=4, freeze-FC attack):**
| scale | TPR | FPR | final acc |
|---|---|---|---|
| s2 | 0.020 | 0.076 | 0.395 ± 0.016 |
| s5 | 0.007 | 0.030 | 0.320 ± 0.001 |

Freeze the FC layer → attacker BER stays at the honest level → MAD evaded (TPR ≈ 0.01)
while the model still collapses to the undefended floor.

**Part 2 (multi-layer FC + `layer1.0.conv2` watermark, MAD α=4, freeze-FC attack):**
| scale | TPR | FPR | att-FC/hon-FC | att-conv/hon-conv | final acc |
|---|---|---|---|---|---|
| s2 | 0.860 ± 0.087 | 0.027 | 0.018/0.021 | 0.201/0.100 | 0.655 ± 0.059 |
| s5 | 0.940 ± 0.040 | 0.017 | 0.028/0.032 | 0.325/0.132 | 0.670 ± 0.026 |

Attacker's FC BER stays frozen-low but its sign-flipped conv BER spikes → detection restored
(TPR 0.02 → 0.86/0.94), accuracy recovered. Layer chosen empirically (Step-1): `layer1.0.conv2`
(18.9 MB P) beats `layer2.1`/`layer3.0` on separation AND memory; honest conv BER erosion grows
with depth (l1 0.17 / l2 0.19 / l3 0.36 vs FC 0.03).

**Escalation (freeze FC + the watermarked conv layer):** TPR 0.020/0.020, acc 0.401/0.318
(s2/s5) — evaded again, because 2 watermarked layers is too little coverage.

**K-scaling coverage curve (escalation, freeze all K watermarked layers, s2, MAD):**

| K (params) | 1 (5K) | 2 (42K) | 3 (78K) | 4 (226K) | 5 (373K) |
|---|---|---|---|---|---|
| final acc | 0.395 | 0.401 | 0.391 | 0.393 | 0.393 |
| honest max-BER | 0.088 | 0.230 | 0.210 | 0.352 | 0.321 |

Coverage payoff is **flat** (acc stays ~0.39 for all K; freezing 373K params still leaves
~97% of ResNet-18 poisonable) while erosion cost **rises** (honest BER 0.09 → 0.35). Per-layer
coverage does not scale.

**SOTA complementarity (freeze-FC attack → final acc, 3 seeds):**

| defense | s2 | s5 |
|---|---|---|
| none | 0.403 ± 0.007 | 0.319 ± 0.005 |
| mad | 0.395 ± 0.016 | 0.320 ± 0.001 |
| krum | 0.689 ± 0.002 | 0.684 ± 0.005 |
| fltrust | 0.678 ± 0.004 | 0.674 ± 0.006 |
| trimmed-mean | 0.665 ± 0.005 | 0.662 ± 0.005 |
| median | 0.675 ± 0.007 | 0.674 ± 0.002 |

`mad` collapses (evaded) while all distance/coordinate baselines hold at clean ~0.68. krum
TPR 1.000 / FPR 0.222. **Central thesis:** the watermark catches what distance methods miss
(Pillar C) and vice-versa (Pillar B).

## Pillar C — BadNets trigger backdoor (final)

Attack: 4×4 white corner patch (`badnets-trigger-*`) on a fraction of attacker images → target 0.
Defenses: `none`, `mad`, `krum`, `fltrust`, 3 seeds. Trigger ASR logged as `server_asr`.

**Stealthy (`badnets-scale=1`, poison 0.5):**

| defense | trigger ASR | main acc | TPR | FPR |
|---|---|---|---|---|
| none | 0.961 ± 0.001 | 0.664 | — | — |
| mad | 0.237 ± 0.097 | 0.696 | 0.913 | 0.059 |
| krum | 0.096 ± 0.001 | 0.688 | — | — |
| fltrust | 0.589 ± 0.018 | 0.675 | — | — |

**Model-replacement (`badnets-scale=10`):**

| defense | trigger ASR | main acc | TPR | FPR |
|---|---|---|---|---|
| none | 0.974 ± 0.001 | 0.707 | — | — |
| mad | 0.095 ± 0.004 | 0.703 | 1.000 | 0.024 |
| krum | 0.093 ± 0.004 | 0.689 | — | — |
| fltrust | 0.572 ± 0.093 | 0.672 | — | — |

Backdoor injects near-perfectly under `none`; `mad` catches both (TPR 0.91/1.00, ASR held to
0.24/0.09). `krum` matches the watermark (ASR ~0.09) — the expected mirror-image (distance
methods *miss* the backdoor) is **not** clean, because the stealthy update is still
*wrong-direction* (the backdoor objective deviates from the honest main-task direction), so
krum rejects it as an outlier. `fltrust` is the weak spot (ASR 0.59/0.57, only partial).

**Constrain-and-scale (`badnets-scale=10`, `badnets-constrain` clips L2 norm; pilot seed 0):**
`pfrac=0.1` is a dead zone — backdoor barely injects (none ASR 0.16–0.32) AND the FC
perturbation is too faint for MAD (att-BER 0.015–0.025 ≈ honest, TPR 0.02–0.22). `pfrac=0.3`
is the candidate sweet spot; full 3-seed matrix not yet run.

## Pillar D — non-IID & attacker-fraction validation (final, MAD α=4)

**Non-IID (Dirichlet):**

| α | setting | final acc | TPR | FPR | att-BER | hon-BER |
|---|---|---|---|---|---|---|
| 0.1 | clean | 0.618 ± 0.000 | — | — | — | 0.161 ± 0.030 |
| 0.1 | s2 | 0.265 ± 0.279 | 0.347 ± 0.397 | 0.009 | 0.419 | 0.349 ± 0.122 |
| 0.5 | clean | 0.668 ± 0.008 | — | — | — | 0.067 ± 0.014 |
| 0.5 | s2 | 0.640 ± 0.017 | 0.873 ± 0.061 | 0.048 | 0.188 | 0.072 |
| 1.0 | clean | 0.687 ± 0.006 | — | — | — | 0.045 ± 0.009 |
| 1.0 | s2 | 0.645 ± 0.039 | 0.847 ± 0.090 | 0.049 | 0.124 | 0.049 |

**Attacker fraction (sign-flip s2, IID):**

| frac | final acc | TPR | FPR |
|---|---|---|---|
| 0.1* (Pillar A) | 0.645 | 0.807 | 0.018 |
| 0.3 | 0.536 ± 0.021 | 0.818 ± 0.032 | 0.010 |
| 0.45 | 0.099 ± 0.009 | 0.075 ± 0.062 | 0.003 |

MAD holds under moderate non-IID (α≥0.5) but degrades under strong non-IID (α=0.1): honest
BER drift widens to 0.16/0.35 and swallows the attacker's 0.42 → TPR ~0.35 (high seed
variance), model collapses. Breakdown point sits between 30% and 45% attackers (TPR 0.82 →
0.08); non-IID itself costs baseline accuracy (0.62–0.69 vs 0.70 IID).

---

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
| `max-trusted-ber` | `0.30` | float | Defense threshold (only used when `defense-method="fixed"`) |
| `defense-method` | `"mad"` | str | `"mad"` (dynamic per-round threshold) or `"fixed"` |
| `mad-alpha` | `3.0` | float | MAD multiplier: Threshold = Median + alpha·MAD |
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
- `run_{timestamp}_pretrain.csv` — epoch, task_loss, watermark_loss, total_loss, watermark_ber
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

## MAD dynamic threshold defense

The defense now defaults to a **per-round dynamic threshold** (`defense-method="mad"`)
instead of the fixed `max-trusted-ber`. Implemented in `watermarked_strategy.py`
(`aggregate_train`), per the user's spec:

1. Collect all local nodes' BER values for the round.
2. `median = median(BERs)`
3. `MAD = median(|BER_i − median|)`
4. `threshold = median + mad-alpha * MAD` (`mad-alpha` default `3.0`)
5. Exclude any node with `BER > threshold` (strictly greater).

Edge cases: `<2` BERs → threshold `+inf` (exclude nobody). `defense-method="fixed"`
restores the old behavior using `max-trusted-ber`. Threshold/median/MAD are logged
each round. New config keys must be added to pyproject.toml before use (flwr
rejects overrides otherwise) and whitelisted in `run_registry.py`.

Note: the fixed-threshold `eval_*` plots and defense scripts still reference the
old BER threshold (0.25) — revisit if/when the MAD defense is evaluated.

## MAD evaluation results (5000/50ep, λ=0.01, 128-bit, 3 reps per config)

`utility scripts/run_defense_mad.sh` + `run_defense_mad_baseline.sh`;
plots via `utility scripts/plot_defense_mad.py` → `results/plots/mad_sweep/`.

Attack sweep (sign_flip, 10% attackers, full 50 FL rounds):

| scale | α | TPR% | FPR% | d' | AUC | final acc% |
|---|---|---|---|---|---|---|
| 5 | 2.0 | 95.3 | 8.2 | 3.25 | 0.982 | 72.5 |
| 5 | 2.5 | 98.7 | 5.0 | 4.15 | 0.995 | 73.4 |
| 5 | 3.0 | 98.7 | 6.9 | 3.78 | 0.993 | 74.0 |
| 2 | 2.0 | 71.3 | 11.1 | 1.60 | 0.868 | 71.2 |
| 2 | 2.5 | 86.0 | 11.2 | 2.39 | 0.931 | 72.5 |
| 2 | 3.0 | 86.0 | 4.3 | 2.48 | 0.944 | 72.4 |

Baseline (NO attacker) FP rate — the reason α barely matters:

| α | FP rounds/50 | FPR/client-round |
|---|---|---|
| 2.0 | 41.3 | 14.4% |
| 2.5 | 30.3 | 9.4% |
| 3.0 | 40.3 | 14.3% |

Key findings:
- Attacker detection near-perfect for scale=5 (only early-round misses); scale=2
  evades in early rounds, worse at α=2.0. Zero misses mid/late.
- Baseline FPs are **systematic**: one partition/run with structurally higher BER
  (different partition each run) is dropped most rounds. Honest BERs quantize at
  1/128, so MAD is tiny (0.004–0.02) vs honest jitter ±0.05 — the threshold sits
  inside the honest noise band and FPR is essentially α-insensitive
  (zero-FP would need α up to ~4 median, ~41 worst).
- α=3.0 (default) is the best single choice: near-best TPR, lowest FPR in attack
  runs, highest final accuracy (~74%, ≈ no-attack baseline).

Same sweep rerun at 5000/30ep (27 runs: 18 attack + 9 baseline) — same picture:

| config | TPR% | FPR% (attack) | FPR% (baseline, no attacker) |
|---|---|---|---|
| scale5 α=2.0 | 99.3 | 7.9 | 11.7 |
| scale5 α=2.5 | 97.3 | 9.6 | 17.3 |
| scale5 α=3.0 | 96.7 | 3.6 | 9.3 |
| scale2 α=2.0 | 83.3 | 13.0 | — |
| scale2 α=2.5 | 68.0 | 9.0 | — |
| scale2 α=3.0 | 84.0 | 7.4 | — |

- 30ep ≈ 50ep qualitatively; final accuracy ~1 pt lower (~71–73%).
- Baseline FPR remains systematic and noisy (±10 rounds/50 across reps); α=3.0
  gives the lowest baseline FPR at both epochs (9.3% at 30ep, 14.3% at 50ep).
- One 30ep run (scale5 α3.0 rep1, `20260806_185440`) hung at round ~48 and wrote
  no CSV; reran manually as `20260807_145812`.

Extended α sweep (4/7/9, 108 total runs: 6 α × {scale5, scale2, baseline} × 3
reps × {30,50}ep). `run_defense_mad_extended.sh`; the first batch was interrupted
by a machine reboot after 8 runs and completed by `run_defense_mad_extended_rest.sh`.

Baseline FPR (no attacker) vs α:

| α | 30ep | 50ep |
|---|---|---|
| 2.0 | 11.7% | 14.4% |
| 2.5 | 17.3% | 9.4% |
| 3.0 | 9.3% | 14.3% |
| 4 | 7.7% | 4.6% |
| 7 | 6.0% | 4.6% |
| 9 | 5.7% | 4.7% |

TPR (sign-flip) vs α:

| α | 30ep s5 | 30ep s2 | 50ep s5 | 50ep s2 |
|---|---|---|---|---|
| 2.0 | 99.3 | 83.3 | 95.3 | 71.3 |
| 3.0 | 96.7 | 84.0 | 98.7 | 86.0 |
| 4 | 95.3 | 74.7 | 97.3 | 74.7 |
| 7 | 86.0 | 62.0 | 90.0 | 38.7 |
| 9 | 46.0 | 61.3 | 81.3 | 48.0 |

- **α=4 is the sweet spot**: baseline FPR drops ~3× (50ep 14.4→4.6%) with
  scale=5 TPR still 95–97% and final acc ~73%. Beyond α=4, FPR plateaus
  (~4.6–5.7%) while detection erodes (50ep scale2 α=7 → TPR 38.7%;
  30ep scale5 α=9 → TPR 46% and accuracy crashes to ~54%).
- So per-node MAD is limited: honest BER spread ≫ MAD (1/128 quantization), so
  even α=9 never reaches zero baseline FP.

## 5000/50 config

- `utility scripts/run_lambda_sweep_50ep.sh` — clean baseline lambda sweep (6 λ × 5 reps = 30 runs)
- `utility scripts/run_defense_50ep.sh` — defense evaluation (1 label_flip + 6 sign_flip × 3 reps = 21 runs)
- Plots: `results/plots/5000_50ep/*.png`
- Results (N=5 sweep, N=3 defense): no BER–λ trend; defense accuracy uniformly higher
  than 30ep (94.0–99.2% vs 89.8–97.8%).

## Pretraining metrics logging

`pretrain_with_watermark` (in `task.py`) returns per-epoch metrics — task loss,
watermark regularization loss, total loss (task + λ·wm), and watermark BER.
`MetricsSaver` writes them to `run_{run_id}_pretrain.csv` (columns:
`epoch, task_loss, watermark_loss, total_loss, watermark_ber`). This lets us
track how the watermark loss evolves during pretraining and how λ drives the
embedding.

`utility scripts/plot_pretrain.py --pretrain-epochs N` plots per-λ mean±std of
watermark loss (log scale), task loss, and BER over pretrain epochs for the
clean baselines of a config, saving to `results/plots/5000_{N}ep/pretrain_evolution.png`.

NOTE: only runs started **after** this logging was added produce
`run_*_pretrain.csv`; existing runs have no pretrain CSV.

## Pretrain-only runs (36) — KEY FINDING

Ran 6 λ × 3 reps × {30,50}ep with `pretrain-only=true` (skips FL rounds;
`utility scripts/run_pretrain_only.sh`, ~2 min each → `pretrain_only.log`).
Plots in `results/plots/5000_{N}ep/pretrain_evolution.png`.

**BER = 0.0 after epoch 1 at EVERY λ (0.005–0.75), both 30ep and 50ep.**
The watermark signs are fully embedded in the first pretrain epoch regardless
of λ; λ only controls the soft margin (watermark-loss decay speed), not the
BER. Mechanism: z = P·W and P·Pᵀ ≈ num_params·I (5120), so the regularization
effective learning rate in watermark-subspace is amplified ~5120×, driving the
signs to b_target almost immediately. This explains why no BER–λ trend was seen
in the defense evals — semifragility comes from FL rounds eroding the watermark,
not from λ-limited embedding during pretraining.

Registry: `log_run` whitelists config keys and initially omitted `pretrain-only`;
the key was added to `run_registry.py` and the 36 run_ids prefixed `20260803`
were backfilled with `"pretrain-only": true` (detected via pretrain CSV existing
with no server CSV). Use `r.get("pretrain-only", False)` to filter them out of
FL-based plots.

## Experiments: 128-bit watermark & pretrain LR scheduler (36 runs)

Two baseline lambda sweeps (6 λ × 3 reps, 50ep/5000 config, full 50 FL rounds,
no attacker), run sequentially (~12 min each, ~7h total) via
`utility scripts/run_exp_128bit.sh` (lr-scheduler sweep script was removed after
the experiment).
Plots: `results/plots/various_experiments/{128bit_watermark,lr_scheduler}/evolution.png`
+ `baseline_lambda_scan.png` per experiment (5-panel evolution + BER/acc vs λ),
generated by
`utility scripts/plot_experiments.py --experiment {128bit,lr_sched} --pretrain-epochs 50`.
The `lr_sched` filter still works on the historical registry data.

Config plumbing added:
- `watermark-num-bits` (whitelisted in registry). NOTE: flwr rejects
  `--run-config` overrides not present in pyproject.toml — any new config key
  MUST be added there first.
- `pretrain-lr-scheduler` was added, tested, and then **removed** (see below).

Findings:
- **128-bit:** pretrain BER still 0 everywhere; wm-loss floor similar to 64-bit.
  FL-end BER still λ-flat (0.094–0.133). Run-to-run variance of FL-end BER DROPS
  vs 64-bit baseline (pooled within-λ std 0.023 vs 0.035, N=3 vs N=5 reps —
  suggestive but reps differ).
- **LR scheduler:** pretrain wm-loss floor collapses ~90× at low λ (λ=0.005:
  0.0045 → 0.00005) and task loss settles lower (~0.001). BUT FL-end BER is NOT
  λ-controlled (0.109–0.154, slightly higher than baseline) and run-to-run
  variance is unchanged (0.034). Tighter pretrain margin does not survive into FL.
- Accuracy ~74% at FL end for all three configs — nothing regressed.
- Overall: neither experiment gives λ control over FL-phase watermark stability;
  BER erosion during FL is driven by FL training, not pretrain margin depth.

## Current default: 128-bit watermark, no LR scheduler

**Decision (2026-08):** `watermark-num-bits = 128` is now the default in
pyproject.toml — the 128-bit redundancy flattens run-to-run variance (kept).
The pretrain LR scheduler was **dropped**: it tightened the pretrain margin
~90× but left FL-end BER, run-to-run variance, and accuracy unchanged, so it is
not worth the config surface. Scheduler code/config was removed from task.py,
server_app.py, pyproject.toml, and the registry whitelist; the lr_sched
experiment data remains in the registry and its plots remain committed.
Future runs use the standard pretrain (no scheduler) with 128 bits by default.
