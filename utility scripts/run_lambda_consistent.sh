#!/bin/bash
# All lambdas at client-samples=4000, 5 reps each
# λ=0.005, 0.05, 0.1, 0.25, 0.75: 2 extra reps (already have 3)
# λ=0.01: 5 fresh reps (had 0 at 4000)
# Total: 15 runs
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/quickstart-pytorch

# 2 extra reps for these (already have 3 from lambda_baseline)
REMAINING=(0.005 0.05 0.1 0.25 0.75)
# 5 fresh reps for lambda=0.01 (original runs were at client-samples=4500)
LAM01=(0.01 0.01 0.01 0.01 0.01)

total=$((${#REMAINING[@]} * 2 + ${#LAM01[@]}))
count=0

for LAM in "${REMAINING[@]}"; do
  for rep in $(seq 1 2); do
    count=$((count + 1))
    echo "============================================"
    echo "  Run $count/$total: λ=$LAM extra rep=$rep"
    echo "============================================"
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
        --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=30 attacker-type=\"none\" watermark-lambda=$LAM" || true
    kill $SUPERLINK_PID 2>/dev/null || true
    wait $SUPERLINK_PID 2>/dev/null || true
    echo "  Done: λ=$LAM extra rep=$rep"
  done
done

for LAM in "${LAM01[@]}"; do
  count=$((count + 1))
  echo "============================================"
  echo "  Run $count/$total: λ=$LAM fresh rep (client-samples=4000)"
  echo "============================================"
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
      --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=30 attacker-type=\"none\" watermark-lambda=$LAM" || true
  kill $SUPERLINK_PID 2>/dev/null || true
  wait $SUPERLINK_PID 2>/dev/null || true
  echo "  Done: λ=$LAM fresh rep"
done

echo "All $total runs completed!"
