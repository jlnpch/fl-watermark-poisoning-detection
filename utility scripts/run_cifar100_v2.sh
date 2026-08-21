#!/bin/bash
# CIFAR-100 VGG-16-BN v2: cosine LR scheduler + features.7.weight watermark layer.
# 1 rep no-WM + 1 rep WM, 200 rounds each.
export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results"

echo "$(date): Starting CIFAR-100 v2 baselines" | tee -a "$LOGDIR/cifar100_v2.log"

# --- Run 1: no-watermark baseline ---
echo "$(date): === CIFAR-100 NO-WATERMARK v2 ===" | tee -a "$LOGDIR/cifar100_v2.log"
bash "$PROJDIR/utility scripts/run_vgg100_v2_nowm.sh" 2>&1 | tee -a "$LOGDIR/cifar100_v2_nowm_full.log"
echo "$(date): === No-WM done ===" | tee -a "$LOGDIR/cifar100_v2.log"

# --- Run 2: watermark baseline ---
echo "$(date): === CIFAR-100 WATERMARK v2 ===" | tee -a "$LOGDIR/cifar100_v2.log"
bash "$PROJDIR/utility scripts/run_vgg100_v2_wm.sh" 2>&1 | tee -a "$LOGDIR/cifar100_v2_wm_full.log"
echo "$(date): === WM done ===" | tee -a "$LOGDIR/cifar100_v2.log"

echo "$(date): ALL CIFAR-100 v2 BASELINES COMPLETE" | tee -a "$LOGDIR/cifar100_v2.log"
