#!/bin/bash
# Re-run all failed runs:
# - Clean baselines missing: λ=0.01×2, 0.1×1, 0.25×2, 0.75×2
# - Defense: label_flip ×1
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/quickstart-pytorch

# Clean baselines (no attack, 5000/30, client-samples=4000)
LAM_MISSING=(0.01 0.01 0.1 0.25 0.25 0.75 0.75)

total=$((${#LAM_MISSING[@]} + 1))  # +1 for label_flip
count=0

for LAM in "${LAM_MISSING[@]}"; do
  count=$((count + 1))
  echo "============================================"
  echo "  Run $count/$total: Clean λ=$LAM (no attack)"
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
  echo "  Done: Clean λ=$LAM"
done

# Label flip defense run
count=$((count + 1))
echo "============================================"
echo "  Run $count/$total: LabelFlip defense"
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
    --run-config 'num-server-rounds=50 server-private-samples=5000 pretrain-epochs=30 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type="iid" local-epochs=1 batch-size=32 attacker-type="label_flip" label-flip-scale=5.0 watermark-lambda=0.01 sign-flip-scale=1.0' || true
kill $SUPERLINK_PID 2>/dev/null || true
wait $SUPERLINK_PID 2>/dev/null || true
echo "  Done: LabelFlip defense"

echo ""
echo "============================================"
echo "  All $total fixes completed!"
echo "============================================"
