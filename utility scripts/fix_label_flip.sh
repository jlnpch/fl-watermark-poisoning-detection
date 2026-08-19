#!/bin/bash
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
    --client-resources-num-gpus 0.1 \
    --init-args-num-gpus 1 2>&1 || true

flwr run /home2/julian/test/fl-watermark-poisoning-detection --stream \
    --run-config 'num-server-rounds=50 server-private-samples=5000 pretrain-epochs=30 attacker-fraction=0.1 max-trusted-ber=0.25 partition-type="iid" local-epochs=1 batch-size=32 attacker-type="label_flip" label-flip-scale=5.0 watermark-lambda=0.01 sign-flip-scale=1.0' || true

kill $SUPERLINK_PID 2>/dev/null || true
wait $SUPERLINK_PID 2>/dev/null || true
echo "Done: label_flip re-run"
