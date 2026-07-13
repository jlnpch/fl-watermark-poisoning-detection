# Results Summary

**Default config:** CIFAR-10, 10 FL nodes, 1 epoch/round, LR=0.01, WD=5e-4, server pretrain 10 epochs on `server-private-samples=5000`, BER threshold defense, 64-bit Uchida watermark in `backbone.fc.weight`. Override any param via `--run-config 'key=val'`.

> **Data exclusivity:** Server takes first `server-private-samples` from the shuffled CIFAR-10 training set; clients receive `client-samples` each from the remaining pool (capped at 50k - server-private-samples total).
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

| Run | λ | Partition | Peak Acc | Best Val Loss | Best R | Rounds | Honest BER |
|---|---|---|---|---|---|---|---|
| — | 0.01 | IID (ES) | 76.17% | 0.7544 | 10 | 20 | max=0.36, mean=0.19 |
| — | 0.00 | IID (ES) | 76.19% | 0.7403 | 10 | 20 | — |
| — | 0.01 | IID | **75.98%** | 0.7564 | — | 50 | mean≈0.19, max≈0.33 |
| — | 0.01 | Dirichlet α=0.5 | 74.72% | 0.8076 | — | 50 | mean≈0.22, max≈0.38 |

BER plot: `results/plots/baseline_noniid_alpha0.5_ber.png`

![non-iid ber](plots/baseline_noniid_alpha0.5_ber.png)

## 2. Noise Attack — With Defense (50 rounds)

| Attack | Peak Acc | Best Val Loss | Att BER | Hon BER (mean / max) | TP | FN | FP |
|---|---|---|---|---|---|---|---|
| noise=0.1 | 75.10% | 0.7710 | 0.44 | 0.14 / 0.27 | 50 | 0 | 0 |
| noise=0.25 | 74.98% | 0.7767 | 0.47 | 0.14 / 0.28 | 50 | 0 | 0 |
| noise=0.5 | 74.94% | 0.7718 | 0.48 | 0.18 / 0.30 | 50 | 0 | 0 |
| noise=1.0 | **75.81%** | 0.7605 | 0.50 | 0.19 / 0.31 | 50 | 0 | 3 |

All at defense threshold θ=0.30.

## 3. Noise Attack — Without Defense (50 rounds)

| Attack | Peak Acc | Best Val Loss | Att BER |
|---|---|---|---|
| noise=0.1 | **76.63%** | 0.7137 | 0.40 |
| noise=0.25 | 68.71% | 0.8902 | 0.49 |
| noise=0.5 | 47.56% | 1.4499 | 0.50 |
| noise=1.0 | 30.13% | 2.1504 | 0.52 |

## 4. Defense Effectiveness

| Noise | Acc (w/o def) | Acc (w/ def) | Δ |
|---|---|---|---|
| 1.0 | 30.13% | 75.81% | **+45.68pp** |
| 0.5 | 47.56% | 74.94% | **+27.38pp** |
| 0.25 | 68.71% | 74.98% | **+6.27pp** |
| 0.1 | 76.63% | 75.10% | −1.53pp |

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
| 10.0 | **76.85%** | 74.95% | −1.90pp | **85.8%** (peak 91.3%) | 0.6% | 0.49 | **50** | 0 | 1 |

Plots: `results/plots/labelflip_sf{1.0,2.0,5.0,10.0}_triple.png`

![lf=1](plots/labelflip_sf1.0_triple.png)
![lf=2](plots/labelflip_sf2.0_triple.png)
![lf=5](plots/labelflip_sf5.0_triple.png)
![lf=10](plots/labelflip_sf10.0_triple.png)

## 7. Label-Flip by Server Private Size (scale=5, with defense)

Different `server-private-samples` values (500, 2500, 5000, 10000) with fixed `client-samples=4000`. Defense threshold = (honest mean BER from no-attack baseline for that size) + 0.15.

### Baseline (no attack) — honest BER means

| Server samples | Honest BER mean | Defense threshold |
|---|---|---|
| 500 | 0.2990 | 0.4490 |
| 2500 | 0.0871 | 0.2371 |
| 5000 | 0.1022 | 0.2522 |
| 10000 | 0.0719 | 0.2219 |

### Label-flip scale=5 results

| Server | Peak Acc | Peak ASR | Last ASR | Hon BER mean | Att BER mean | TP | FP |
|---|---|---|---|---|---|---|---|
| 500 | 74.53% | **65.2%** | 5.5% | 0.3525 | 0.4091 | 11 | 1 |
| 2500 | 72.97% | 1.3% | 0.5% | 0.1499 | 0.4437 | **50** | 2 |
| 5000 | 73.50% | 1.3% | 0.8% | 0.1043 | 0.4781 | **50** | 3 |
| 10000 | **76.74%** | 0.9% | 0.2% | **0.0491** | 0.4738 | **50** | 5 |

Plots: `results/plots/labelflip_server{500,2500,5000,10000}_{ber,acc}.png`

**Observations:**
- **server=500**: defense threshold is very high (0.449) because baseline honest BER is already 0.30 — the watermark barely embeds with so little pretraining. The attacker's BER (0.41) is barely above threshold, so only 11/50 rounds are caught. ASR peaks at 65.2%.
- **server=2500/5000/10000**: with reasonable watermark embedding, the attacker is consistently detected (TP=50/50). The backdoor is suppressed effectively (peak ASR ≤1.3%).
- **Honest BER decreases** as server private size increases (0.35 → 0.05), making the defense more precise.
- The **threshold + 0.15** heuristic works well: it avoids false positives while catching the attacker, except when the baseline BER is already high (server=500).

## 8. Observations (legacy experiments — default 5000 samples)

### Label-Flip scale sweep (default 5000 samples)

- Pure backdoor (scale=1.0): weak ASR (~2.8%), watermark barely disturbed (att BER=0.18), near-invisible to defense.
- Scale 2.0–5.0: amplification helps the backdoor propagate (ASR 4.5–5.5%) but **also distorts the watermark** (att BER 0.40–0.45), making the attacker trivially detectable — TP=50/50 at both scales.
- **Scale 10.0 finally breaks through:** ASR reaches 85.8% (peak 91.3%) without defense. Accuracy stays high (76.85%). However the watermark defense still catches every round (TP=50/50, att BER=0.49) and suppresses ASR to 0.6%. The backdoor is effective but completely detectable by the watermark.
