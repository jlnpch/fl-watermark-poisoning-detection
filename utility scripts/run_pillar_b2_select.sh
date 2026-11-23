#!/bin/bash
# Pillar B Part 2, Step 1: empirical conv-layer selection for multi-layer watermark.
#
# For FC + each candidate conv layer, measure over 50 FL rounds:
#   (a) honest BER drift (clean run, no attack)
#   (b) freeze-FC sign-flip (s2) attacker BER spike
#
# Candidates (ResNet-18 conv2 weights): layer1.0 (18.9MB P), layer2.1 (75.5MB),
# layer3.0 (302MB). All with defense=none (attacker spike is client-side).
#
# Matrix: 3 candidates x {clean, freeze_fc s2} = 6 runs, seed 0.
#
# Launch detached:
#   setsid bash "utility scripts/run_pillar_b2_select.sh" > /tmp/pillar_b2_select_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_b2_select_summary.log"

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 5
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

run_one() {
    local tag="$1" wl="$2" attack="$3" frac="$4" scale="$5" seed="$6"
    local logfile="$LOGDIR/pillar_b2_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (layers=$wl attack=$attack) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc=""
    if [ "$attack" = "clean" ]; then
        rc="seed=${seed} defense-method=\"none\" attacker-type=\"none\" attacker-fraction=0.0 watermark-layers=\"${wl}\""
    else
        rc="seed=${seed} defense-method=\"none\" attacker-type=\"freeze_fc\" attacker-fraction=${frac} sign-flip-scale=${scale} frozen-layers=\"backbone.fc.weight\" watermark-layers=\"${wl}\""
    fi

    flwr run "$PROJDIR" --stream --run-config "$rc" > "$logfile" 2>&1 || true

    local RUNID=""
    RUNID=$(grep -oE 'Run ID: [0-9_]+' "$logfile" | head -1 | awk '{print $3}')

    kill "$SUPERLINK_PID" 2>/dev/null || true
    wait "$SUPERLINK_PID" 2>/dev/null || true

    echo "=== $(date '+%F %T') END   $tag (run_id=$RUNID) ==="
    printf '%s\tdefense=none\tattack=%s\tlayers=%s\tscale=%s\tseed=%s\trun_id=%s\n' \
        "$tag" "$attack" "$wl" "$scale" "$seed" "$RUNID" >> "$SUMMARY"
}

FC="backbone.fc.weight"
C1="backbone.layer1.0.conv2.weight"
C2="backbone.layer2.1.conv2.weight"
C3="backbone.layer3.0.conv2.weight"

# Clean runs (honest BER drift)
run_one "sel_clean_l1_seed0" "${FC},${C1}" "clean" "0.0" "none" "0"
run_one "sel_clean_l2_seed0" "${FC},${C2}" "clean" "0.0" "none" "0"
run_one "sel_clean_l3_seed0" "${FC},${C3}" "clean" "0.0" "none" "0"

# freeze-FC s2 runs (attacker spike)
run_one "sel_fcfc_s2_l1_seed0" "${FC},${C1}" "freeze_fc" "0.1" "2.0" "0"
run_one "sel_fcfc_s2_l2_seed0" "${FC},${C2}" "freeze_fc" "0.1" "2.0" "0"
run_one "sel_fcfc_s2_l3_seed0" "${FC},${C3}" "freeze_fc" "0.1" "2.0" "0"

echo "$(date '+%F %T') PILLAR B2 SELECTION COMPLETE (6 runs)" >> "$SUMMARY"
