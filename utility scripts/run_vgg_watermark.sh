#!/bin/bash
# VGG-16 watermark baseline: watermark (lambda=0.01), no attack.
# Pretrain with watermark regularization, then FL — confirms watermark survives training.
# 50 FL rounds, 10 clients, 3 repetitions.
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

REPETITIONS=3

for rep in $(seq 1 $REPETITIONS); do
  echo ""
  echo "============================================"
  echo "  VGG-16 watermark baseline (no attack) rep=$rep/$REPETITIONS"
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
      --client-resources-num-gpus 0.3 \
      --init-args-num-gpus 1 2>&1 || true

  flwr run /home2/julian/test/fl-watermark-poisoning-detection --stream \
      --run-config 'model-name="vgg16" num-classes=10 dataset-name="uoft-cs/cifar10" num-server-rounds=50 server-private-samples=5000 pretrain-epochs=50 watermark-lambda=0.01 watermark-num-bits=128 watermark-layer="classifier.6.weight" attacker-type="none" defense-method="none"' || true

  kill $SUPERLINK_PID 2>/dev/null || true
  wait $SUPERLINK_PID 2>/dev/null || true

  echo ""
  echo "  Done: VGG-16 watermark baseline rep=$rep"
  echo ""
done

echo ""
echo "============================================"
echo "  All $REPETITIONS VGG-16 watermark baseline runs completed!"
echo "============================================"
