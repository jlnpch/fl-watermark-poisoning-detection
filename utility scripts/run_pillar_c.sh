#!/bin/bash
# Pillar C — targeted backdoor (BadNets) vs watermark + distance baselines.
#
# Two variants:
#   stealthy          : badnets-scale=1.0  (normal-magnitude update)
#   model-replacement : badnets-scale=10.0 (aggregate ≈ attacker model)
# Defenses: none, mad, krum, fltrust. 3 seeds. 2 x 4 x 3 = 24 runs.
#
# Launch detached:
#   setsid bash "utility scripts/run_pillar_c.sh" > /tmp/pillar_c_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_c_summary.log"
FAILED="$LOGDIR/pillar_c_failed.log"

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 6
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
    sleep 3
}

run_one() {
    local tag="$1" defense="$2" scale="$3" seed="$4"
    local logfile="$LOGDIR/pillar_c_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (def=$defense badnets-scale=$scale seed=$seed) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc="seed=${seed} defense-method=\"${defense}\" mad-alpha=4.0 attacker-type=\"badnets\" attacker-fraction=0.1 badnets-scale=${scale} badnets-target=0 badnets-trigger-size=4 badnets-trigger-value=1.0 badnets-trigger-pos=\"bottom-right\" badnets-poison-frac=0.5"

    timeout -k 60 1800 flwr run "$PROJDIR" --stream --run-config "$rc" > "$logfile" 2>&1
    local rc_code=$?

    local RUNID=""
    RUNID=$(grep -oE 'Run ID: [0-9_]+' "$logfile" | head -1 | awk '{print $3}')

    kill "$SUPERLINK_PID" 2>/dev/null || true
    wait "$SUPERLINK_PID" 2>/dev/null || true

    if [ $rc_code -ne 0 ]; then
        echo "=== $(date '+%F %T') FAIL $tag (exit=$rc_code run_id=$RUNID) ==="
        printf '%s\tdefense=%s\tscale=%s\tseed=%s\texit=%s\trun_id=%s\n' \
            "$tag" "$defense" "$scale" "$seed" "$rc_code" "$RUNID" >> "$FAILED"
    else
        echo "=== $(date '+%F %T') END   $tag (run_id=$RUNID) ==="
        printf '%s\tdefense=%s\tattack=badnets\tbadnets-scale=%s\tseed=%s\trun_id=%s\n' \
            "$tag" "$defense" "$scale" "$seed" "$RUNID" >> "$SUMMARY"
    fi
}

for variant in "stealthy:1.0" "replace:10.0"; do
    VNAME="${variant%%:*}"; VSCALE="${variant#*:}"
    for defense in none mad krum fltrust; do
        for seed in 0 1 2; do
            run_one "pillarc_${VNAME}_${defense}_seed${seed}" "$defense" "$VSCALE" "$seed"
        done
    done
done

echo "$(date '+%F %T') PILLAR C COMPLETE" >> "$SUMMARY"
