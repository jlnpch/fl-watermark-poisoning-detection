# ResNet-18 Results Summary

**Config:** CIFAR-10, 10 FL nodes, 1 epoch/round, LR=0.01, WD=5e-4, server pretrain 10 epochs, BER threshold defense

| Run | Attack | Defense | Peak Acc | Best Val Loss | Best R | Early Stop | TP | FP |
|---|---|---|---|---|---|---|---|---|
| 1 | None | — | **74.89%** | 0.7898 | 9 | 19 | — | — |
| 2 | scale=2.0 | thresh=0.35 | 74.71% | 0.7905 | 9 | 19 | 7/19 | 0 |
| 3 | scale=2.0 | thresh=0.30 | 73.48% | 0.7823 | 7 | 17 | 7/17 | 1 |
| 4 | scale=5.0 | none | 70.86% | 0.8615 | 10 | 20 | — | — |
| 5 | scale=5.0 | thresh=0.35 | 74.83% | **0.7787** | 8 | 18 | **18/18** | **0** |
| 6 | scale=10.0 | none | **74.82%** | 0.8477 | 33 | 43 | — | — |
