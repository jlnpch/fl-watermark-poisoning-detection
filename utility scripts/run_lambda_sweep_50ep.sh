#!/bin/bash
# Clean baseline lambda sweep for 5000/50 config (no attack, 4000 client samples)
# 6 lambdas × 5 reps = 30 runs
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/quickstart-pytorch

LAMBDA_VALUES=(0.005 0.01 0.05 0.1 0.25 0.75)
REPETITIONS=5

total=$((${#LAMBDA_VALUES[@]} * REPETITIONS))
count=0

for LAM in "${LAMBDA_VALUES[@]}"; do
  for rep in $(seq 1 $REPETITIONS); do
    count=$((count + 1))
    echo ""
    echo "============================================"
    echo "  Run $count/$total: λ=$LAM rep=$rep (no attack, 50 pretrain epochs)"
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

    flwr run /home2/julian/test/quickstart-pytorch --stream \
        --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-type=\"none\" watermark-lambda=$LAM" || true

    kill $SUPERLINK_PID 2>/dev/null || true
    wait $SUPERLINK_PID 2>/dev/null || true

    echo ""
    echo "  Done: λ=$LAM rep=$rep"
    echo ""
  done
done

echo ""
echo "============================================"
echo "  All $total lambda sweep runs completed!"
echo "============================================"
