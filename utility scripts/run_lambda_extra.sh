#!/bin/bash
# Additional lambda sweep: 0.25 and 0.75, 3 reps each = 6 runs
# Fixed: 5000 server samples, 30 pretrain epochs, 50 FL rounds,
#        10 supernodes, sign_flip scale=5, BER threshold 0.25

export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

LAMBDA_VALUES=(0.25 0.75)
REPETITIONS=3

total=$((${#LAMBDA_VALUES[@]} * REPETITIONS))
count=0

for LAM in "${LAMBDA_VALUES[@]}"; do
  for rep in $(seq 1 $REPETITIONS); do
    count=$((count + 1))
    echo ""
    echo "============================================"
    echo "  Run $count/$total: SignFlip λ=$LAM rep=$rep"
    echo "============================================"
    echo ""

    # Cleanup stale processes
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 5
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    sleep 3

    # Start superlink
    flower-superlink --insecure --simulation &
    SUPERLINK_PID=$!
    sleep 5

    # Configure federation
    flwr federation simulation-config @none/default local \
        --num-supernodes 10 \
        --client-resources-num-gpus 0.1 \
        --init-args-num-gpus 1 2>&1 || true

    # Run experiment
    flwr run /home2/julian/test/fl-watermark-poisoning-detection --stream \
        --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=30 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"sign_flip\" sign-flip-scale=5.0 watermark-lambda=$LAM label-flip-scale=1.0" || true

    # Kill superlink
    kill $SUPERLINK_PID 2>/dev/null || true
    wait $SUPERLINK_PID 2>/dev/null || true

    echo ""
    echo "  Done: SignFlip λ=$LAM rep=$rep"
    echo ""
  done
done

echo ""
echo "============================================"
echo "  All $total additional runs completed!"
echo "============================================"
