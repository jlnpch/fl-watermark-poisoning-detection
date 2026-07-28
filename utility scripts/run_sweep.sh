#!/bin/bash
# Baseline sweep: server_private_samples × pretrain_epochs × 3 repetitions
# Total: 4 × 3 × 3 = 36 runs

export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/quickstart-pytorch

SERVER_SIZES=(500 2500 5000 10000)
PRETRAIN_EPOCHS=(10 30 50)
REPETITIONS=3
NUM_ROUNDS=50

total=$((${#SERVER_SIZES[@]} * ${#PRETRAIN_EPOCHS[@]} * REPETITIONS))
count=0

for samples in "${SERVER_SIZES[@]}"; do
  for epochs in "${PRETRAIN_EPOCHS[@]}"; do
    for rep in $(seq 1 $REPETITIONS); do
      count=$((count + 1))
      echo ""
      echo "============================================"
      echo "  Run $count/$total: samples=$samples epochs=$epochs rep=$rep"
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
      flwr run /home2/julian/test/quickstart-pytorch --stream \
          --run-config "num-server-rounds=$NUM_ROUNDS server-private-samples=$samples pretrain-epochs=$epochs attacker-type=\"none\" partition-type=\"iid\"" || true

      # Kill superlink
      kill $SUPERLINK_PID 2>/dev/null || true
      wait $SUPERLINK_PID 2>/dev/null || true

      echo ""
      echo "  Done: samples=$samples epochs=$epochs rep=$rep"
      echo ""
    done
  done
done

echo ""
echo "============================================"
echo "  All $total runs completed!"
echo "============================================"
