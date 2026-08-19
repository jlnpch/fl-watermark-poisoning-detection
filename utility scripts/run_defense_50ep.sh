#!/bin/bash
# Semi-fragility defense evaluation for 5000/50 config
# 1 label_flip + 6 sign_flip = 7 configs × 3 reps = 21 runs
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

REPETITIONS=3

CONFIGS=(
  "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"label_flip\" label-flip-scale=5.0 watermark-lambda=0.01 sign-flip-scale=1.0"
  "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"sign_flip\" sign-flip-scale=5.0 watermark-lambda=0.005 label-flip-scale=1.0"
  "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"sign_flip\" sign-flip-scale=5.0 watermark-lambda=0.01 label-flip-scale=1.0"
  "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"sign_flip\" sign-flip-scale=5.0 watermark-lambda=0.05 label-flip-scale=1.0"
  "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"sign_flip\" sign-flip-scale=5.0 watermark-lambda=0.10 label-flip-scale=1.0"
  "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"sign_flip\" sign-flip-scale=5.0 watermark-lambda=0.25 label-flip-scale=1.0"
  "num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type=\"iid\" local-epochs=1 batch-size=32 attacker-type=\"sign_flip\" sign-flip-scale=5.0 watermark-lambda=0.75 label-flip-scale=1.0"
)

total=$((${#CONFIGS[@]} * REPETITIONS))
count=0

LABELS=(
  "LabelFlip_scale5|label_flip|0.01"
  "SignFlip_scale5_lam0005|sign_flip|0.005"
  "SignFlip_scale5_lam001|sign_flip|0.01"
  "SignFlip_scale5_lam005|sign_flip|0.05"
  "SignFlip_scale5_lam010|sign_flip|0.10"
  "SignFlip_scale5_lam025|sign_flip|0.25"
  "SignFlip_scale5_lam075|sign_flip|0.75"
)

for i in $(seq 0 $((${#CONFIGS[@]} - 1))); do
  IFS='|' read -r name attack lam <<< "${LABELS[$i]}"
  for rep in $(seq 1 $REPETITIONS); do
    count=$((count + 1))
    echo ""
    echo "============================================"
    echo "  Run $count/$total: $name rep=$rep (attack=$attack lam=$lam, 50 pretrain epochs)"
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
        --run-config "${CONFIGS[$i]}" || true

    kill $SUPERLINK_PID 2>/dev/null || true
    wait $SUPERLINK_PID 2>/dev/null || true

    echo ""
    echo "  Done: $name rep=$rep"
    echo ""
  done
done

echo ""
echo "============================================"
echo "  All $total defense evaluation runs completed!"
echo "============================================"
