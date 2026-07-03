# ResNet-18 Results Summary

**Config:** CIFAR-10, 10 FL nodes, 1 epoch/round, LR=0.01, WD=5e-4, server pretrain 10 epochs, BER threshold defense

## Label-Shift Attack (labels+1) + Gradient Scaling

| Run | Attack | Defense | Peak Acc | Best Val Loss | Best R | Early Stop | TP | FP |
|---|---|---|---|---|---|---|---|---|
| 1 | None | — | **74.89%** | 0.7898 | 9 | 19 | — | — |
| 2 | scale=2.0 | thresh=0.35 | 74.71% | 0.7905 | 9 | 19 | 7/19 | 0 |
| 3 | scale=2.0 | thresh=0.30 | 73.48% | 0.7823 | 7 | 17 | 7/17 | 1 |
| 4 | scale=5.0 | none | 70.86% | 0.8615 | 10 | 20 | — | — |
| 5 | scale=5.0 | thresh=0.35 | 74.83% | **0.7787** | 8 | 18 | 18/18 | **0** |
| 6 | scale=10.0 | none | **74.82%** | 0.8477 | 33 | 43 | — | — |

## Noise Attack (replace update with `N(0, noise_scale)`)

| Run | Attack | Defense | Peak Acc | Best Val Loss | Best R | Early Stop | TP | FN | FP |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | None | — | 74.89% | 0.7898 | 9 | 19 | — | — | — |
| 7 | noise=1.0 | thresh=0.35 | 74.12% | 0.7865 | 9 | 19 | 18 | 1 | 0 |
| 8 | noise=0.5 | thresh=0.30 | 74.92% | 0.7820 | 8 | 18 | 18 | 0 | 5 |
| 9 | noise=0.1 | thresh=0.30 | **74.99%** | 0.7679 | 9 | 18 | 19 | 0 | 0 |
| 10 | noise=0.05 | thresh=0.30 | **75.64%** | **0.7640** | 15 | 19 | 12 | 7 | 1 |
