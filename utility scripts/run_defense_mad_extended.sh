#!/bin/bash
# Extended MAD alpha sweep: alpha 4/7/9 at BOTH 5000/30ep and 5000/50ep.
# Attack (sign_flip scale 5 & 2) + baseline (no attacker), 3 reps each.
# Totals: attack 2 scales x 3 alphas x 3 reps x 2 epochs = 36
#         baseline 3 alphas x 3 reps x 2 epochs = 18
#         total = 54 runs
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

EPOCHS=(30 50)
SCALES=(5 2)
ALPHAS=(4 7 9)
REPETITIONS=3

total=54
count=0

for EPOCHS_VAL in "${EPOCHS[@]}"; do
  for SCALE in "${SCALES[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
      for rep in $(seq 1 $REPETITIONS); do
        count=$((count + 1))
        echo ""
        echo "============================================"
        echo "  Run $count/$total: ATTACK epochs=$EPOCHS_VAL scale=$SCALE alpha=$ALPHA rep=$rep"
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
            --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=$EPOCHS_VAL watermark-lambda=0.01 watermark-num-bits=128 attacker-type=\"sign_flip\" attacker-fraction=0.1 sign-flip-scale=$SCALE defense-method=\"mad\" mad-alpha=$ALPHA" || true

        kill $SUPERLINK_PID 2>/dev/null || true
        wait $SUPERLINK_PID 2>/dev/null || true

        echo ""
        echo "  Done: ATTACK epochs=$EPOCHS_VAL scale=$SCALE alpha=$ALPHA rep=$rep"
        echo ""
      done
    done
  done

  for ALPHA in "${ALPHAS[@]}"; do
    for rep in $(seq 1 $REPETITIONS); do
      count=$((count + 1))
      echo ""
      echo "============================================"
      echo "  Run $count/$total: BASELINE epochs=$EPOCHS_VAL alpha=$ALPHA rep=$rep"
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
          --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=$EPOCHS_VAL watermark-lambda=0.01 watermark-num-bits=128 attacker-type=\"none\" attacker-fraction=0 defense-method=\"mad\" mad-alpha=$ALPHA" || true

      kill $SUPERLINK_PID 2>/dev/null || true
      wait $SUPERLINK_PID 2>/dev/null || true

      echo ""
      echo "  Done: BASELINE epochs=$EPOCHS_VAL alpha=$ALPHA rep=$rep"
      echo ""
    done
  done
done

echo ""
echo "============================================"
echo "  All $total extended runs completed!"
echo "============================================"
