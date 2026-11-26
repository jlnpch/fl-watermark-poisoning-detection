#!/bin/bash
# Pillar C — constrain-and-scale backdoor (Bagdasaryan et al.).
#
# The attacker scales its backdoor update by badnets-scale=10 (model replacement),
# then clips the L2 norm to `badnets-constrain * ||raw_update||` so the sent update
# keeps the backdoor DIRECTION but has benign MAGNITUDE. Combined with a small
# badnets-poison-frac (subtle direction), this is the attempt to evade distance
# methods (krum/fltrust) while still perturbing the FC layer (watermark catches).
#
# Matrix: poison-frac x constrain x {none,mad,krum,fltrust} x seeds.
#   POISON_FRACS=(0.1 0.3)  CONSTRAINS=(2.0 4.0)
#   pilot = seed 0 only (16 runs); full = 3 seeds (48 runs).
#
# Usage:
#   setsid bash "utility scripts/run_pillar_c_constrain.sh" pilot > /tmp/pillar_c_cs_master.log 2>&1 &
#   setsid bash "utility scripts/run_pillar_c_constrain.sh" full  > /tmp/pillar_c_cs_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_c_constrain_summary.log"
FAILED="$LOGDIR/pillar_c_constrain_failed.log"

MODE="${1:-pilot}"
POISON_FRACS=(0.1 0.3)
CONSTRAINS=(2.0 4.0)
DEFENSES=(none mad krum fltrust)
SCALE=10.0

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 6
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 $pid 2>/dev/null; done
    sleep 3
}

run_one() {
    local tag="$1" defense="$2" pfrac="$3" constrain="$4" seed="$5"
    local logfile="$LOGDIR/pillar_c_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (def=$defense pfrac=$pfrac constrain=$constrain seed=$seed) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc="seed=${seed} defense-method=\"${defense}\" mad-alpha=4.0 attacker-type=\"badnets\" attacker-fraction=0.1 badnets-scale=${SCALE} badnets-constrain=${constrain} badnets-poison-frac=${pfrac} badnets-target=0 badnets-trigger-size=4 badnets-trigger-value=1.0 badnets-trigger-pos=\"bottom-right\""

    timeout -k 60 1800 flwr run "$PROJDIR" --stream --run-config "$rc" > "$logfile" 2>&1
    local rc_code=$?

    local RUNID=""
    RUNID=$(grep -oE 'Run ID: [0-9_]+' "$logfile" | head -1 | awk '{print $3}')

    kill "$SUPERLINK_PID" 2>/dev/null || true
    wait "$SUPERLINK_PID" 2>/dev/null || true

    if [ $rc_code -ne 0 ]; then
        echo "=== $(date '+%F %T') FAIL $tag (exit=$rc_code run_id=$RUNID) ==="
        printf '%s\tdefense=%s\tpfrac=%s\tconstrain=%s\tseed=%s\texit=%s\trun_id=%s\n' \
            "$tag" "$defense" "$pfrac" "$constrain" "$seed" "$rc_code" "$RUNID" >> "$FAILED"
    else
        echo "=== $(date '+%F %T') END   $tag (run_id=$RUNID) ==="
        printf '%s\tdefense=%s\tattack=badnets\tpoison-frac=%s\tconstrain=%s\tscale=%s\tseed=%s\trun_id=%s\n' \
            "$tag" "$defense" "$pfrac" "$constrain" "$SCALE" "$seed" "$RUNID" >> "$SUMMARY"
    fi
}

SEEDS=(0)
[ "$MODE" = "full" ] && SEEDS=(0 1 2)

for pfrac in "${POISON_FRACS[@]}"; do
    for constrain in "${CONSTRAINS[@]}"; do
        for defense in "${DEFENSES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                tag="cs_p${pfrac}_c${constrain}_${defense}_seed${seed}"
                run_one "$tag" "$defense" "$pfrac" "$constrain" "$seed"
            done
        done
    done
done

echo "$(date '+%F %T') PILLAR C CONSTRAIN-AND-SCALE COMPLETE ($MODE)" >> "$SUMMARY"
