# Results Summary

**Default config:** CIFAR-10, 10 FL nodes, 1 epoch/round, LR=0.01, WD=5e-4, server pretrain 10 epochs on exclusive 10% data, BER threshold defense, 64-bit Uchida watermark in `backbone.fc.weight`. Override any param via `--run-config 'key=val'`.

> **Data exclusivity:** Server takes first 10% of CIFAR-10 exclusively; clients partitioned from remaining 90% via `Dataset.shard(contiguous=True)`.
>
> **Run config reference:** See [`pyproject.toml`](pyproject.toml) `[tool.flwr.app.config]` for all parameters.
>
> **CSV output:** After every run, per-round per-client metrics land in `results/run_{timestamp}_{server,train,eval}.csv`.
> Use `pandas.read_csv()` to load for plotting.

---

## Usage

```bash
# No-attacker baseline
flwr run . --stream --run-config 'attacker-type=none watermark-lambda=0.01'

# Noise attack (attacker-type=noise)
flwr run . --stream \
  --run-config 'attacker-type=noise attacker-fraction=0.1 attacker-noise-scale=0.5 \
                max-trusted-ber=0.30 watermark-lambda=0.01'

# Label flip with gradient scaling (attacker-type=label_flip)
flwr run . --stream \
  --run-config 'attacker-type=label_flip attacker-fraction=0.1 label-flip-offset=1 \
                label-flip-scale=5.0 max-trusted-ber=0.35 watermark-lambda=0.01'

# No watermark (λ=0)
flwr run . --stream \
  --run-config 'attacker-type=none watermark-lambda=0.0 max-trusted-ber=1.0'
```

---

## 1. No-Attacker Baselines

| Run | λ | Peak Acc | Best Val Loss | Best R | ES Round | Honest BER |
|---|---|---|---|---|---|---|
| — | 0.01 | **76.17%** | **0.7544** | 10 | 20 | max=0.36, mean=0.19 |
| — | 0.00 | **76.19%** | 0.7403 | 10 | 20 | — |

## 2. Noise Attack — With Defense

| Run | Attack | Defense | Peak Acc | Best Val Loss | Best R | ES | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| — | noise=1.0 | thresh=0.35 | 74.77% | 0.7908 | 9 | 19 | 19 | 0 | 0 |
| — | noise=0.5 | thresh=0.30 | 75.12% | 0.7741 | 7 | 17 | 17 | 0 | 4 |
| — | noise=0.25 | thresh=0.30 | **75.07%** | 0.7729 | 9 | 19 | 19 | 0 | 0 |
| — | noise=0.1 | thresh=0.30 | **75.29%** | 0.7816 | 8 | 18 | 18 | 0 | 0 |

Attacker BER: noise=1.0 → ~0.50 (random), noise=0.5 → ~0.49, noise=0.1 → ~0.47.
Honest BER: all conditions ~0.15–0.19 mean, max ≤ 0.36.

## 3. Noise Attack — Without Defense

| Run | Attack | Peak Acc | Best Val Loss | Rounds |
|---|---|---|---|---|
| — | noise=1.0 | **23.93%** | 2.5392 | 16 |
| — | noise=0.5 | **47.03%** | 1.4631 | 44 |
| — | noise=0.25 | **70.41%** | 0.8420 | 100 |
| — | noise=0.1 | **75.91%** | 0.7250 | 38 |

## 4. Defense Effectiveness

| Noise | Acc (w/o def) | Acc (w/ def) | Δ |
|---|---|---|---|
| 1.0 | 23.93% | 74.77% | **+50.84pp** |
| 0.5 | 47.03% | 75.12% | **+28.09pp** |
| 0.25 | 70.41% | 75.07% | **+4.66pp** |
| 0.1 | 75.91% | 75.29% | −0.62pp |

## 5. Sign-Flip Attack

Attacker sends `initial_state − scale × (trained − initial_state)`. At scale=1.0 the update is small (per-round drift is tiny), so attacker BER stays low and detection fails. At scale ≥ 2 the inversion is strong enough to raise BER and degrade accuracy.

| Scale | Acc (no def) | Acc (w/ def) | Δ | Att BER | TP | FN | FP |
|---|---|---|---|---|---|---|---|
| 1.0 | 73.09% | 73.27% | +0.18pp | 0.15 | 0 | 50 | 1 |
| 2.0 | 70.19% | 72.58% | **+2.39pp** | 0.25 | 6 | 44 | 3 |
| 5.0 | 60.54% | 72.12% | **+11.58pp** | 0.26 | 9 | 41 | 0 |

Plots: `results/plots/signflip_sf{1,2,5}.png`

![sf=1](plots/signflip_sf1.png)
![sf=2](plots/signflip_sf2.png)
![sf=5](plots/signflip_sf5.png)

**Observations:**
- Scale 1.0: update vector is small → flipped model stays near the (watermarked) initial state → BER barely rises above honest baseline → defense sees nothing.
- Scale 2.0: degradation starts (−3pp without defense), BER crosses threshold in ~12% of rounds.
- Scale 5.0: strong degradation (−15.6pp without defense), defense recovers +11.58pp but still misses 41/50 attacker rounds — honest BER also climbs because the corrupted global model degrades everyone's watermark.
