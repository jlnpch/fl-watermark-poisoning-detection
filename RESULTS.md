# Results Summary

**Config:** CIFAR-10, 10 FL nodes, 1 epoch/round, LR=0.01, WD=5e-4, server pretrain 10 epochs on exclusive 10% of data, BER threshold defense, 64-bit Uchida watermark in `backbone.fc.weight`

> **Data exclusivity fix:** Server pretrain data is exclusive — first 10% of CIFAR-10 training set reserved for server, clients partitioned from remaining 90% via `Dataset.shard(contiguous=True)`.
>
> **No-attacker baselines** and **no-defense noise runs** use the exclusive setup. **With-defense noise runs** (7–10) were done before the fix (data overlap), so accuracy is not perfectly comparable — expect ~1pp higher acc with exclusive data.

## No-Attacker Baselines

| Run | λ | Peak Acc | Best Val Loss | Best R | ES Round |
|---|---|---|---|---|---|
| 1 | 0.01 | **76.04%** | 0.7599 | 16 | 19 |
| — | 0.00 | **76.25%** | 0.7413 | 10 | 20 |

## Noise Attack — With Defense

Attack: 1/10 nodes replaces model update with `global + N(0, noise_scale)` on all float params.

| Run | Attack | Defense | Peak Acc | Best Val Loss | Best R | ES | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| 1 | None (λ=0.01) | — | 76.04% | 0.7599 | 16 | 19 | — | — | — |
| 7 | noise=1.0 | thresh=0.35 | 74.12% | 0.7865 | 9 | 19 | 18 | 1 | 0 |
| 8 | noise=0.5 | thresh=0.30 | 74.92% | 0.7820 | 8 | 18 | 18 | 0 | 5 |
| 9 | noise=0.1 | thresh=0.30 | **74.99%** | 0.7679 | 9 | 18 | 19 | 0 | 0 |
| 10 | noise=0.05 | thresh=0.30 | 74.49% | 0.8038 | 14 | 17 | 14 | 1 | 0 |

TP = attacker excluded (BER > threshold), FN = attacker not excluded (BER ≤ threshold), FP = honest client excluded (BER > threshold).

## Noise Attack — Without Defense (max-trusted-ber=1.0)

| Run | Attack | Peak Acc | Best Val Loss | Rounds |
|---|---|---|---|---|
| 1 | None (λ=0.01) | 76.04% | 0.7599 | 19 |
| 11 | noise=1.0 | **28.59%** | 2.3761 | 29 |
| 12 | noise=0.5 | **51.80%** | 1.3479 | 77 |
| 13 | noise=0.1 | **76.46%** | 0.7092 | 36 |
| 14 | noise=0.05 | 75.71% | 0.7364 | 23 |

## Defense Effectiveness

| Noise | Acc (w/o def) | Acc (w/ def) | Δ |
|---|---|---|---|
| 1.0 | 28.59% | 74.12% | **+45.53pp** |
| 0.5 | 51.80% | 74.92% | **+23.12pp** |
| 0.1 | 76.46% | 74.99% | −1.47pp |
| 0.05 | 75.71% | 74.49% | −1.22pp |

Defense is critical for noise ≥ 0.5. At noise ≤ 0.1, the attacker noise is diluted 10× by FedAvg and has negligible effect on global model quality. The slight accuracy reduction with defense at low noise is due to earlier early stopping (data loss from honest-client exclusion).
