#!/bin/bash
# Pillar D — non-IID and attacker-fraction validation (quick overview).
#
# Locked baseline: ResNet-18/CIFAR-10, 128-bit Uchida FC watermark, MAD a=4.0,
# 50 rounds, 10 clients, 3 seeds. Defense = mad only.
#
# Experiment 1 (non-IID): partition-type=dirichlet, partition-alpha in {0.1,0.5,1.0}
#   x {clean, sign_flip s2 (frac 0.1)} = 18 runs.
# Experiment 2 (attacker fraction / breakdown): sign_flip s2, attacker-fraction in
#   {0.3, 0.45} (the 0.1 cell is Pillar A's sign_flip s2 + mad) = 6 runs.
#
# Launch detached:
#   setsid bash "utility scripts/run_pillar_d.sh" > /tmp/pillar_d_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_d_summary.log"
FAILED="$LOGDIR/pillar_d_failed.log"

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 6
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
    sleep 3
}

run_one() {
    local tag="$1" defense="$2" attack="$3" frac="$4" scale="$5" seed="$6" ptype="$7" palpha="$8"
    local logfile="$LOGDIR/pillar_d_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (atk=$attack frac=$frac seed=$seed ptype=$ptype alpha=$palpha) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc="seed=${seed} defense-method=\"${defense}\" mad-alpha=4.0 attacker-type=\"${attack}\" attacker-fraction=${frac} sign-flip-scale=${scale} partition-type=\"${ptype}\" partition-alpha=${palpha}"

    timeout -k 60 1800 flwr run "$PROJDIR" --stream --run-config "$rc" > "$logfile" 2>&1
    local rc_code=$?

    local RUNID=""
    RUNID=$(grep -oE 'Run ID: [0-9_]+' "$logfile" | head -1 | awk '{print $3}')

    kill "$SUPERLINK_PID" 2>/dev/null || true
    wait "$SUPERLINK_PID" 2>/dev/null || true

    if [ $rc_code -ne 0 ]; then
        echo "=== $(date '+%F %T') FAIL $tag (exit=$rc_code run_id=$RUNID) ==="
        printf '%s\tdefense=%s\tattack=%s\tfrac=%s\tseed=%s\tptype=%s\talpha=%s\texit=%s\trun_id=%s\n' \
            "$tag" "$defense" "$attack" "$frac" "$seed" "$ptype" "$palpha" "$rc_code" "$RUNID" >> "$FAILED"
    else
        echo "=== $(date '+%F %T') END   $tag (run_id=$RUNID) ==="
        printf '%s\tdefense=%s\tattack=%s\tattacker-fraction=%s\tseed=%s\tpartition-type=%s\tpartition-alpha=%s\trun_id=%s\n' \
            "$tag" "$defense" "$attack" "$frac" "$seed" "$ptype" "$palpha" "$RUNID" >> "$SUMMARY"
    fi
}

# ---------------- Experiment 1: non-IID (Dirichlet) ----------------
for alpha in 0.1 0.5 1.0; do
    for seed in 0 1 2; do
        run_one "d1_dirichlet_a${alpha}_clean_seed${seed}" "mad" "none" "0.0" "2.0" "$seed" "dirichlet" "$alpha"
    done
    for seed in 0 1 2; do
        run_one "d1_dirichlet_a${alpha}_s2_seed${seed}" "mad" "sign_flip" "0.1" "2.0" "$seed" "dirichlet" "$alpha"
    done
done

# ---------------- Experiment 2: attacker fraction (breakdown) ----------------
for frac in 0.3 0.45; do
    for seed in 0 1 2; do
        run_one "d2_frac${frac}_s2_seed${seed}" "mad" "sign_flip" "$frac" "2.0" "$seed" "iid" "0.5"
    done
done

echo "$(date '+%F %T') PILLAR D COMPLETE (24 runs)" >> "$SUMMARY"
