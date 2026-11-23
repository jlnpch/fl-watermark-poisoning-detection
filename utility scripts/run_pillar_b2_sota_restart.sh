#!/bin/bash
# Pillar B SOTA comparison — RESTART of the remaining 23 runs (the original sweep
# hung on krum s2 seed1 due to a Ray/fork deadlock).
#
# Remaining: krum (s2 seed1,2 + s5 x3), fltrust x6, trimmed-mean x6, median x6.
# Each run wrapped in `timeout` so a hang can't stall the sweep; failures logged.
#
# Launch detached:
#   setsid bash "utility scripts/run_pillar_b2_sota_restart.sh" > /tmp/pillar_b2_sota_restart_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_b2_sota_restart_summary.log"
FAILED="$LOGDIR/pillar_b2_sota_failed.log"

FC="backbone.fc.weight"

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 6
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
    sleep 3
}

run_one() {
    local tag="$1" defense="$2" scale="$3" seed="$4"
    local logfile="$LOGDIR/pillar_b2_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (def=$defense s=$scale seed=$seed) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc="seed=${seed} defense-method=\"${defense}\" mad-alpha=4.0 attacker-type=\"freeze_fc\" attacker-fraction=0.1 sign-flip-scale=${scale} frozen-layers=\"${FC}\" watermark-layers=\"${FC}\""

    # 30 min hard timeout; kill-after 60s if TERM is ignored (fork deadlock).
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
        printf '%s\tdefense=%s\tattack=freeze_fc\tscale=%s\tseed=%s\trun_id=%s\n' \
            "$tag" "$defense" "$scale" "$seed" "$RUNID" >> "$SUMMARY"
    fi
}

for defense in krum fltrust trimmed-mean median; do
    for scale in 2.0 5.0; do
        for seed in 0 1 2; do
            if [ "$defense" = "krum" ] && [ "$scale" = "2.0" ] && [ "$seed" = "0" ]; then
                continue  # already completed
            fi
            run_one "sota_${defense}_s${scale}_seed${seed}" "$defense" "$scale" "$seed"
        done
    done
done

echo "$(date '+%F %T') PILLAR B2 SOTA RESTART COMPLETE" >> "$SUMMARY"
