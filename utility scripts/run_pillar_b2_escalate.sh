#!/bin/bash
# Pillar B Part 2, Step 4: escalation — attacker freezes FC AND the watermarked
# conv layer (coverage argument). Multi-layer MAD (alpha=4.0).
#
# Matrix: sign-flip scales {2.0, 5.0} x seeds {0,1,2} = 6 runs.
#
# Launch detached:
#   setsid bash "utility scripts/run_pillar_b2_escalate.sh" > /tmp/pillar_b2_escalate_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_b2_escalate_summary.log"

WL="backbone.fc.weight,backbone.layer1.0.conv2.weight"
FROZEN="backbone.fc.weight,backbone.layer1.0.conv2.weight"

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 5
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

run_one() {
    local tag="$1" scale="$2" seed="$3"
    local logfile="$LOGDIR/pillar_b2_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (escalation freeze FC+conv s=$scale seed=$seed) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc="seed=${seed} defense-method=\"mad\" mad-alpha=4.0 attacker-type=\"freeze_fc\" attacker-fraction=0.1 sign-flip-scale=${scale} frozen-layers=\"${FROZEN}\" watermark-layers=\"${WL}\""

    flwr run "$PROJDIR" --stream --run-config "$rc" > "$logfile" 2>&1 || true

    local RUNID=""
    RUNID=$(grep -oE 'Run ID: [0-9_]+' "$logfile" | head -1 | awk '{print $3}')

    kill "$SUPERLINK_PID" 2>/dev/null || true
    wait "$SUPERLINK_PID" 2>/dev/null || true

    echo "=== $(date '+%F %T') END   $tag (run_id=$RUNID) ==="
    printf '%s\tdefense=mad\tattack=freeze_fc\tlayers=%s\tfrozen=%s\tscale=%s\tseed=%s\trun_id=%s\n' \
        "$tag" "$WL" "$FROZEN" "$scale" "$seed" "$RUNID" >> "$SUMMARY"
}

for scale in 2.0 5.0; do
    for seed in 0 1 2; do
        run_one "esc_s${scale}_seed${seed}" "$scale" "$seed"
    done
done

echo "$(date '+%F %T') PILLAR B2 ESCALATION COMPLETE (6 runs)" >> "$SUMMARY"
