#!/bin/bash
# Pillar A: comparison against SOTA Byzantine-robust aggregation baselines.
#
# Matrix: 6 defenses x 4 attack settings x 3 seeds = 72 runs.
#   defenses: none, mad, fltrust, krum, trimmed-mean, median
#   attacks : clean (no attack), sign_flip scale 2.0, sign_flip scale 5.0, noise scale 1.0
#   seeds   : 0, 1, 2
#
# Usage:
#   bash "utility scripts/run_pillar_a.sh" pilot   # sign_flip s2 x 6 defenses x seed 0 (6 runs)
#   bash "utility scripts/run_pillar_a.sh" full    # all 72 runs
#
# Launch detached with setsid so it survives disconnection:
#   setsid bash "utility scripts/run_pillar_a.sh" full > /tmp/pillar_a_master.log 2>&1 &

export PATH="/home/julian/.local/bin:$PATH"
PROJDIR="/home2/julian/test/fl-watermark-poisoning-detection"
LOGDIR="$PROJDIR/results/logs"
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/pillar_a_summary.log"

MODE="${1:-pilot}"

DEFENSES=(none mad fltrust krum trimmed-mean median)

cleanup() {
    kill -9 $(ps aux | grep -E "ray|flwr" | grep -v grep | awk '{print $2}') 2>/dev/null || true
    sleep 5
    nvidia-smi | grep python | awk '{print $5}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

run_one() {
    local tag="$1" defense="$2" attack="$3" frac="$4" scale="$5" seed="$6"
    local logfile="$LOGDIR/pillar_a_${tag}.log"

    echo "=== $(date '+%F %T') START $tag (defense=$defense attack=$attack scale=$scale seed=$seed) ==="

    cleanup

    flower-superlink --insecure --simulation > /dev/null 2>&1 &
    local SUPERLINK_PID=$!
    sleep 5

    flwr federation simulation-config @none/default local \
        --num-supernodes 10 --client-resources-num-gpus 0.1 --init-args-num-gpus 1 2>&1 || true

    local rc=""
    case "$attack" in
        clean)
            rc="seed=${seed} defense-method=\"${defense}\" attacker-type=\"none\" attacker-fraction=0.0"
            ;;
        sign_flip)
            rc="seed=${seed} defense-method=\"${defense}\" attacker-type=\"sign_flip\" attacker-fraction=${frac} sign-flip-scale=${scale}"
            ;;
        noise)
            rc="seed=${seed} defense-method=\"${defense}\" attacker-type=\"noise\" attacker-fraction=${frac} attacker-noise-scale=${scale}"
            ;;
    esac

    flwr run "$PROJDIR" --stream --run-config "$rc" > "$logfile" 2>&1 || true

    local RUNID=""
    RUNID=$(grep -oE 'Run ID: [0-9_]+' "$logfile" | head -1 | awk '{print $3}')

    kill "$SUPERLINK_PID" 2>/dev/null || true
    wait "$SUPERLINK_PID" 2>/dev/null || true

    echo "=== $(date '+%F %T') END   $tag (run_id=$RUNID) ==="
    printf '%s\tdefense=%s\tattack=%s\tscale=%s\tseed=%s\trun_id=%s\n' \
        "$tag" "$defense" "$attack" "$scale" "$seed" "$RUNID" >> "$SUMMARY"
}

# --- Pilot: sign_flip scale 2.0, seed 0, all six defenses ---
if [ "$MODE" = "pilot" ]; then
    for defense in "${DEFENSES[@]}"; do
        run_one "s2_${defense}_seed0" "$defense" "sign_flip" "0.1" "2.0" "0"
    done
    echo "$(date '+%F %T') PILOT COMPLETE (6 runs)" >> "$SUMMARY"
    exit 0
fi

# --- Full 72-run sweep ---
# Order: clean, then sign_flip s2, s5, then noise. Within each attack: defense x seed.
for defense in "${DEFENSES[@]}"; do
    for seed in 0 1 2; do
        run_one "clean_${defense}_seed${seed}" "$defense" "clean" "0.0" "none" "$seed"
    done
done

for defense in "${DEFENSES[@]}"; do
    for seed in 0 1 2; do
        run_one "s2_${defense}_seed${seed}" "$defense" "sign_flip" "0.1" "2.0" "$seed"
    done
done

for defense in "${DEFENSES[@]}"; do
    for seed in 0 1 2; do
        run_one "s5_${defense}_seed${seed}" "$defense" "sign_flip" "0.1" "5.0" "$seed"
    done
done

for defense in "${DEFENSES[@]}"; do
    for seed in 0 1 2; do
        run_one "noise_${defense}_seed${seed}" "$defense" "noise" "0.1" "1.0" "$seed"
    done
done

echo "$(date '+%F %T') FULL SWEEP COMPLETE (72 runs)" >> "$SUMMARY"
