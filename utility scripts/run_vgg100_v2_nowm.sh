#!/bin/bash
# CIFAR-100 VGG-16-BN v2: no-watermark, cosine LR, 200 rounds.
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
    --run-config 'model-name="vgg16" num-classes=100 dataset-name="uoft-cs/cifar100" num-server-rounds=200 server-private-samples=5000 pretrain-epochs=50 watermark-lambda=0 watermark-num-bits=128 attacker-type="none" defense-method="none" lr-scheduler="cosine" lr-min=0.0001' || true

kill $SUPERLINK_PID 2>/dev/null || true
wait $SUPERLINK_PID 2>/dev/null || true

echo ""
echo "  Done: CIFAR-100 v2 no-watermark"
