#!/bin/bash
# MAD-defense BASELINE runs (no attacker): measure FP rate of the dynamic
# threshold when all clients are honest. 3 alphas x 3 reps = 9 runs.
# Config: 50 pretrain epochs, 5000 server samples, watermark-lambda=0.01, 128-bit.
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

ALPHAS=(2.0 2.5 3.0)
REPETITIONS=3

total=$(( ${#ALPHAS[@]} * REPETITIONS ))
count=0

for ALPHA in "${ALPHAS[@]}"; do
  for rep in $(seq 1 $REPETITIONS); do
    count=$((count + 1))
    echo ""
    echo "============================================"
    echo "  Run $count/$total: baseline (no attack) alpha=$ALPHA rep=$rep"
    echo "============================================"
    echo ""

    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 5
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    sleep 3

    flower-superlink --insecure --simulation &
    SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 \
        --client-resources-num-gpus 0.1 \
        --init-args-num-gpus 1 2>&1 || true

    flwr run /home2/julian/test/fl-watermark-poisoning-detection --stream \
        --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 watermark-lambda=0.01 watermark-num-bits=128 attacker-type=\"none\" attacker-fraction=0 defense-method=\"mad\" mad-alpha=$ALPHA" || true

    kill $SUPERLINK_PID 2>/dev/null || true
    wait $SUPERLINK_PID 2>/dev/null || true

    echo ""
    echo "  Done: baseline alpha=$ALPHA rep=$rep"
    echo ""
  done
done

echo ""
echo "============================================"
echo "  All $total baseline runs completed!"
echo "============================================"
