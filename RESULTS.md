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

# Label flip backdoor (attacker-type=label_flip)
flwr run . --stream \
  --run-config 'attacker-type=label_flip attacker-fraction=0.1 label-flip-source=9 \
                label-flip-target=2 label-flip-scale=1.0 max-trusted-ber=0.35 watermark-lambda=0.01'

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
| 5.0 (50r) | 60.54% | 72.12% (th=0.30) | **+11.58pp** | 0.26 | 9 | 41 | 0 |
| 5.0 (100r) | 59.48% | 71.14% (th=0.30) | **+11.66pp** | 0.28 | 48 | 52 | 1 |
| 5.0 (100r) | 59.48% | **72.08%** (th=0.25) | **+12.60pp** | 0.27 | 58 | 42 | 77 |

Plots: `results/plots/signflip_sf{1.0,2.0}_triple.png` + `signflip_sf5.0_th{0.30,0.25}_triple.png`

![sf=1](plots/signflip_sf1.0_triple.png)
![sf=2](plots/signflip_sf2.0_triple.png)
![sf=5 th=0.30](plots/signflip_sf5.0_th0.30_triple.png)
![sf=5 th=0.25](plots/signflip_sf5.0_th0.25_triple.png)

**Observations:**
- Scale 1.0: update vector is small → flipped model stays near the (watermarked) initial state → BER barely rises above honest baseline → defense sees nothing.
- Scale 2.0: degradation starts (−3pp without defense), BER crosses threshold in ~12% of rounds.
- Scale 5.0: strong degradation (−15.6pp without defense), defense recovers +11.58pp (50r, th=0.30) / +11.66pp (100r, th=0.30) / +12.60pp (100r, th=0.25). Over 100 rounds the attacker crosses threshold more often (48/100 TP vs 9/50). Lowering threshold to 0.25 boosts TP to 58 but causes 77 false positives (honest BER spikes get excluded), though accuracy still improves slightly (72.08% vs 71.14%). See both threshold variants in the plots above.

## 6. Label-Flip Backdoor Attack

The malicious client replaces source-class labels with target-class at the Dataset level before training (no loss-function or gradient manipulation). `label-flip-scale` additionally amplifies the model update residual. ASR measures the fraction of source-class test images predicted as the target class by the global model.

| Scale | Acc (no def) | Acc (w/ def) | Δ | ASR (no def) | ASR (w/ def) | Att BER | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|
| 1.0 | 75.10% | 75.64% | +0.54pp | 2.8% | 2.8% | 0.18 | 6 | 44 | 2 |
| 2.0 | 75.45% | 75.08% | −0.37pp | 4.5% | **0.9%** | 0.40 | **50** | 0 | 10 |
| 5.0 | 75.40% | 74.66% | −0.74pp | 5.5% | **0.8%** | 0.45 | **50** | 0 | 0 |

Plots: `results/plots/labelflip_sf{1.0,2.0,5.0}_triple.png`

![lf=1](plots/labelflip_sf1.0_triple.png)
![lf=2](plots/labelflip_sf2.0_triple.png)
![lf=5](plots/labelflip_sf5.0_triple.png)

**Observations:**
- Pure backdoor (scale=1.0): weak ASR (~2.8%), watermark barely disturbed (att BER=0.18), near-invisible to defense.
- Scale 2.0–5.0: amplification helps the backdoor propagate (ASR 4.5–5.5%) but **also distorts the watermark** (att BER 0.40–0.45), making the attacker trivially detectable — TP=50/50 at both scales.
- The watermark defense effectively neutralizes the amplified backdoor: ASR drops from ~5% to under 1%, with negligible accuracy cost (≤0.74pp drop vs no-def).
