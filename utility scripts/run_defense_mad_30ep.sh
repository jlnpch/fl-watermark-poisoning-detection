#!/bin/bash
# MAD-defense test vs sign-flip attack, 30 pretrain epochs (5000/30ep) (scale 5 and 2), sweeping alpha (2.0/2.5/3.0).
# Config: 30 pretrain epochs, 5000 server samples, watermark-lambda=0.01, 128-bit.
# 2 scales x 3 alphas x 3 reps = 18 runs, full 50 FL rounds.
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

SCALES=(5 2)
ALPHAS=(2.0 2.5 3.0)
REPETITIONS=3

total=$(( ${#SCALES[@]} * ${#ALPHAS[@]} * REPETITIONS ))
count=0

for SCALE in "${SCALES[@]}"; do
  for ALPHA in "${ALPHAS[@]}"; do
    for rep in $(seq 1 $REPETITIONS); do
      count=$((count + 1))
      echo ""
      echo "============================================"
      echo "  Run $count/$total: sign_flip scale=$SCALE alpha=$ALPHA rep=$rep"
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
          --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=30 watermark-lambda=0.01 watermark-num-bits=128 attacker-type=\"sign_flip\" attacker-fraction=0.1 sign-flip-scale=$SCALE defense-method=\"mad\" mad-alpha=$ALPHA" || true

      kill $SUPERLINK_PID 2>/dev/null || true
      wait $SUPERLINK_PID 2>/dev/null || true

      echo ""
      echo "  Done: sign_flip scale=$SCALE alpha=$ALPHA rep=$rep"
      echo ""
    done
  done
done

echo ""
echo "============================================"
echo "  All $total MAD-defense runs completed!"
echo "============================================"
