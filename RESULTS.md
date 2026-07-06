# Results Summary

**Default config:** CIFAR-10, 10 FL nodes, 1 epoch/round, LR=0.01, WD=5e-4, server pretrain 10 epochs on exclusive 10% data, BER threshold defense, 64-bit Uchida watermark in `backbone.fc.weight`. Override any param via `--run-config 'key=val'`.

> **Data exclusivity:** Server takes first 10% of CIFAR-10 exclusively; clients partitioned from remaining 90% via `Dataset.shard(contiguous=True)`.
>
> **Run config reference:** See [`pyproject.toml`](pyproject.toml) `[tool.flwr.app.config]` for all parameters.

---

## Template: Adding a New Experiment

```bash
# Noise attack (attacker-type=noise)
flwr run . --stream \
  --run-config 'attacker-type=noise attacker-fraction=0.1 attacker-noise-scale=0.5 \
                max-trusted-ber=0.30 watermark-lambda=0.01'

# Label-shift + gradient scaling (attacker-type=combined)
flwr run . --stream \
  --run-config 'attacker-type=combined attacker-fraction=0.1 label-shift-offset=1 \
                gradient-scale=5.0 max-trusted-ber=0.35 watermark-lambda=0.01'

# No-attacker baseline
flwr run . --stream \
  --run-config 'attacker-type=none watermark-lambda=0.01'

# No watermark (λ=0)
flwr run . --stream \
  --run-config 'attacker-type=none watermark-lambda=0.0 max-trusted-ber=1.0'
```

Copy a row template below and paste at the end of the relevant section.

> **Row template (with defense):**
> `| RUN | type=X [params] | thresh=Y | PEAK | LOSS | R | ES | TP | FN | FP |`
>
> **Row template (without defense / baseline):**
> `| RUN | type=X [params] | PEAK | LOSS | R |`

---

## 1. No-Attacker Baselines

| Run | λ | Peak Acc | Best Val Loss | Best R | ES Round |
|---|---|---|---|---|---|
| 1 | 0.01 | **76.04%** | 0.7599 | 16 | 19 |
| — | 0.00 | **76.25%** | 0.7413 | 10 | 20 |

## 2. Noise Attack — With Defense

| Run | Attack | Defense | Peak Acc | Best Val Loss | Best R | ES | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| 1 | None (λ=0.01) | — | 76.04% | 0.7599 | 16 | 19 | — | — | — |
| 7 | noise=1.0 | thresh=0.35 | 74.12% | 0.7865 | 9 | 19 | 18 | 1 | 0 |
| 8 | noise=0.5 | thresh=0.30 | 74.92% | 0.7820 | 8 | 18 | 18 | 0 | 5 |
| 9 | noise=0.1 | thresh=0.30 | **74.99%** | 0.7679 | 9 | 18 | 19 | 0 | 0 |
| 10 | noise=0.05 | thresh=0.30 | 74.49% | 0.8038 | 14 | 17 | 14 | 1 | 0 |

## 3. Noise Attack — Without Defense

| Run | Attack | Peak Acc | Best Val Loss | Rounds |
|---|---|---|---|---|
| 1 | None (λ=0.01) | 76.04% | 0.7599 | 19 |
| 11 | noise=1.0 | **28.59%** | 2.3761 | 29 |
| 12 | noise=0.5 | **51.80%** | 1.3479 | 77 |
| 13 | noise=0.1 | **76.46%** | 0.7092 | 36 |
| 14 | noise=0.05 | 75.71% | 0.7364 | 23 |

## 4. Defense Effectiveness

| Noise | Acc (w/o def) | Acc (w/ def) | Δ |
|---|---|---|---|
| 1.0 | 28.59% | 74.12% | **+45.53pp** |
| 0.5 | 51.80% | 74.92% | **+23.12pp** |
| 0.1 | 76.46% | 74.99% | −1.47pp |
| 0.05 | 75.71% | 74.49% | −1.22pp |

Defense is critical for noise ≥ 0.5. At noise ≤ 0.1, the 10× FedAvg dilution makes the attacker's noise negligible.
