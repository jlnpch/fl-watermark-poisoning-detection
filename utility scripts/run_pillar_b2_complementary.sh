#!/bin/bash
# Pillar B complementary runs (2 phases).
#
# Phase 1 — K-scaling coverage curve: escalate (freeze ALL K watermarked layers)
#   under MAD (a=4.0), sign-flip s2, 3 seeds, for K=3,4,5 (K=1,2 already done).
#   Yields coverage payoff (final acc) vs erosion (honest max-BER) vs K.
#
# Phase 2 — SOTA comparison: freeze-FC attacker (freezes FC only) against the
#   distance/coordinate baselines {none, krum, fltrust, trimmed-mean, median},
#   sign-flip s2 & s5, 3 seeds. Shows the complementarity: watermark (mad)
#   collapses, distance methods hold.
#
# Launch detached:
#   setsid bash "utility scripts/run_pillar_b2_complementary.sh" > /tmp/pillar_b2_comp_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_b2_complementary_summary.log"

FC="backbone.fc.weight"
L10="backbone.layer1.0.conv2.weight"
L11="backbone.layer1.1.conv2.weight"
L20="backbone.layer2.0.conv2.weight"
L21="backbone.layer2.1.conv2.weight"

K3="${FC},${L10},${L11}"
K4="${FC},${L10},${L11},${L20}"
K5="${FC},${L10},${L11},${L20},${L21}"

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 5
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

run_one() {
    local tag="$1" defense="$2" attack="$3" frac="$4" scale="$5" seed="$6" wl="$7" frozen="$8"
    local logfile="$LOGDIR/pillar_b2_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (def=$defense atk=$attack s=$scale seed=$seed) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc="seed=${seed} defense-method=\"${defense}\" mad-alpha=4.0 attacker-type=\"${attack}\" attacker-fraction=${frac} sign-flip-scale=${scale}"
    [ -n "$wl" ] && rc="$rc watermark-layers=\"${wl}\""
    [ -n "$frozen" ] && rc="$rc frozen-layers=\"${frozen}\""

    flwr run "$PROJDIR" --stream --run-config "$rc" > "$logfile" 2>&1 || true

    local RUNID=""
    RUNID=$(grep -oE 'Run ID: [0-9_]+' "$logfile" | head -1 | awk '{print $3}')

    kill "$SUPERLINK_PID" 2>/dev/null || true
    wait "$SUPERLINK_PID" 2>/dev/null || true

    echo "=== $(date '+%F %T') END   $tag (run_id=$RUNID) ==="
    printf '%s\tdefense=%s\tattack=%s\tscale=%s\tseed=%s\tlayers=%s\tfrozen=%s\trun_id=%s\n' \
        "$tag" "$defense" "$attack" "$scale" "$seed" "$wl" "$frozen" "$RUNID" >> "$SUMMARY"
}

# ---------------- Phase 1: K-scaling coverage curve ----------------
for kspec in "3:${K3}" "4:${K4}" "5:${K5}"; do
    K="${kspec%%:*}"; WL="${kspec#*:}"
    for seed in 0 1 2; do
        run_one "kscale_K${K}_seed${seed}" "mad" "freeze_fc" "0.1" "2.0" "$seed" "$WL" "$WL"
    done
done

# ---------------- Phase 2: SOTA comparison (freeze-FC) ----------------
for defense in none krum fltrust trimmed-mean median; do
    for scale in 2.0 5.0; do
        for seed in 0 1 2; do
            run_one "sota_${defense}_s${scale}_seed${seed}" "$defense" "freeze_fc" "0.1" "$scale" "$seed" "$FC" "$FC"
        done
    done
done

echo "$(date '+%F %T') PILLAR B2 COMPLEMENTARY COMPLETE" >> "$SUMMARY"
