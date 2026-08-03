#!/bin/bash
# Pretrain-only runs: 6 lambdas × 3 reps for BOTH 30ep and 50ep configs (36 runs)
# Skips federated rounds (pretrain-only=true) — only the pretrain CSV is produced.
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/quickstart-pytorch

LAMBDA_VALUES=(0.005 0.01 0.05 0.1 0.25 0.75)
CONFIGS=(30 50)
REPETITIONS=3

total=$((${#LAMBDA_VALUES[@]} * ${#CONFIGS[@]} * REPETITIONS))
count=0

for EP in "${CONFIGS[@]}"; do
  for LAM in "${LAMBDA_VALUES[@]}"; do
    for rep in $(seq 1 $REPETITIONS); do
      count=$((count + 1))
      echo ""
      echo "============================================"
      echo "  Run $count/$total: ${EP}ep λ=$LAM rep=$rep (pretrain-only)"
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
          --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=$EP attacker-type=\"none\" pretrain-only=true watermark-lambda=$LAM" || true

      kill $SUPERLINK_PID 2>/dev/null || true
      wait $SUPERLINK_PID 2>/dev/null || true

      echo ""
      echo "  Done: ${EP}ep λ=$LAM rep=$rep"
      echo ""
    done
  done
done

echo ""
echo "============================================"
echo "  All $total pretrain-only runs completed!"
echo "============================================"
