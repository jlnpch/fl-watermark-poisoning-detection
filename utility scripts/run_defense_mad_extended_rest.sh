#!/bin/bash
# Completion of the extended MAD alpha sweep (alpha 4/7/9) that was interrupted
# by a machine reboot on 2026-08-07. Runs the 46 missing configs:
#   - 30ep scale5 alpha9 rep3 (1, rep already complete)
#   - 30ep scale2 alpha4/7/9 x3 reps (9)
#   - 30ep baseline alpha4/7/9 x3 reps (9)
#   - 50ep attack scale5+scale2 alpha4/7/9 x3 reps (18)
#   - 50ep baseline alpha4/7/9 x3 reps (9)
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

# format: epochs|scale_or_BASELINE|alpha|reps
CONFIGS=(
  "30|5|9|1"
  "30|2|4|3"
  "30|2|7|3"
  "30|2|9|3"
  "30|BASELINE|4|3"
  "30|BASELINE|7|3"
  "30|BASELINE|9|3"
  "50|5|4|3"
  "50|5|7|3"
  "50|5|9|3"
  "50|2|4|3"
  "50|2|7|3"
  "50|2|9|3"
  "50|BASELINE|4|3"
  "50|BASELINE|7|3"
  "50|BASELINE|9|3"
)

total=0
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r EP SC AL REPS <<< "$c"
  total=$((total + REPS))
done
count=0

for c in "${CONFIGS[@]}"; do
  IFS='|' read -r EP SC AL REPS <<< "$c"
  for rep in $(seq 1 $REPS); do
    count=$((count + 1))
    echo ""
    echo "============================================"
    echo "  Run $count/$total: epochs=$EP scale=$SC alpha=$AL rep=$rep"
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

    if [ "$SC" = "BASELINE" ]; then
      flwr run /home2/julian/test/fl-watermark-poisoning-detection --stream \
          --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=$EP watermark-lambda=0.01 watermark-num-bits=128 attacker-type=\"none\" attacker-fraction=0 defense-method=\"mad\" mad-alpha=$AL" || true
    else
      flwr run /home2/julian/test/fl-watermark-poisoning-detection --stream \
          --run-config "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=$EP watermark-lambda=0.01 watermark-num-bits=128 attacker-type=\"sign_flip\" attacker-fraction=0.1 sign-flip-scale=$SC defense-method=\"mad\" mad-alpha=$AL" || true
    fi

    kill $SUPERLINK_PID 2>/dev/null || true
    wait $SUPERLINK_PID 2>/dev/null || true

    echo ""
    echo "  Done: epochs=$EP scale=$SC alpha=$AL rep=$rep"
    echo ""
  done
done

echo ""
echo "============================================"
echo "  All $total completion runs finished!"
echo "============================================"
