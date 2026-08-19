#!/bin/bash
# VGG-16-BN watermark baseline on CIFAR-100: watermark (lambda=0.01), no attack.
# Pretrain with watermark regularization, then FL 200 rounds, 10 clients, 1 rep.
# GPU fraction 0.3 → max 3 concurrent actors.
export PATH="/home/julian/.local/bin:$PATH"
cd /home2/julian/test/fl-watermark-poisoning-detection

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
    --run-config 'model-name="vgg16" num-classes=100 dataset-name="uoft-cs/cifar100" num-server-rounds=200 server-private-samples=5000 pretrain-epochs=50 watermark-lambda=0.01 watermark-num-bits=128 watermark-layer="classifier.6.weight" attacker-type="none" defense-method="none"' || true

kill $SUPERLINK_PID 2>/dev/null || true
wait $SUPERLINK_PID 2>/dev/null || true

echo ""
echo "  Done: VGG-16-BN CIFAR-100 watermark baseline"
