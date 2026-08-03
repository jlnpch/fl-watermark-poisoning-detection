#!/usr/bin/env python3
"""Plot pretraining evolution: watermark/task loss + BER over pretrain epochs.

For each λ (clean baseline, 5000/{ep} config) it averages the per-epoch
metrics from `run_*_pretrain.csv` over available repetitions and draws:
  1. watermark regularization loss  vs pretrain epoch
  2. task (primal) loss             vs pretrain epoch
  3. watermark BER                  vs pretrain epoch

Output: results/plots/5000_{ep}ep/pretrain_evolution.png
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
LAMBDAS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.75]


def parse_args():
    parser = argparse.ArgumentParser(description="Pretraining metrics plots")
    parser.add_argument("--pretrain-epochs", type=int, default=30,
                        help="Pretrain epochs of the config (default: 30)")
    return parser.parse_args()


def load_registry():
    reg_path = RESULTS_DIR / "run_registry.jsonl"
    if not reg_path.exists():
        print(f"ERROR: {reg_path} not found")
        sys.exit(1)
    runs = []
    with open(reg_path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def load_pretrain_metrics(run_id):
    """Return dict: metric_name -> list over epochs, or None if missing."""
    path = RESULTS_DIR / f"run_{run_id}_pretrain.csv"
    if not path.exists():
        return None
    cols = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            for key in ("epoch", "task_loss", "watermark_loss", "total_loss", "watermark_ber"):
                if key in row:
                    cols[key].append(float(row[key]))
    return dict(cols)


def interpolate(common_epochs, rep_data):
    result = []
    for d in rep_data:
        rmap = dict(zip(d["epoch"], d))
        result.append({m: np.array([d[m][d["epoch"].index(e)] for e in common_epochs]) for m in
                       ("task_loss", "watermark_loss", "total_loss", "watermark_ber")})
    return result


def main():
    args = parse_args()
    ep = args.pretrain_epochs
    runs = load_registry()
    print(f"Found {len(runs)} runs in registry")

    config_dir = RESULTS_DIR / "plots" / f"5000_{ep}ep"
    config_dir.mkdir(parents=True, exist_ok=True)

    # Clean baselines per λ (same filter as the lambda scan)
    lam_runs = {}
    for lam in LAMBDAS:
        matches = [r for r in runs
                   if r.get("attacker-type", "none") == "none"
                   and r.get("attacker-fraction", 0) == 0
                   and r.get("server-private-samples") == 5000
                   and r.get("pretrain-epochs") == ep
                   and r.get("client-samples") == 4000
                   and abs(r.get("watermark-lambda", 0) - lam) < 1e-9]
        # dedup by run_id, keep runs that actually produced a pretrain CSV
        seen = set()
        lam_runs[lam] = []
        for r in matches:
            rid = r["run_id"]
            if rid in seen:
                continue
            seen.add(rid)
            if load_pretrain_metrics(rid):
                lam_runs[lam].append(r)

    # Only lambdas with at least one pretrain CSV
    active = [lam for lam in LAMBDAS if lam_runs[lam]]
    if not active:
        print("\nNo pretrain CSVs found yet for the %dep config." % ep)
        print("Runs started after the pretrain-logging change produce run_*_pretrain.csv.")
        sys.exit(1)

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(active)))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    metrics = [("watermark_loss", "Watermark regularization loss", "C3"),
               ("task_loss", "Task (primal) loss", "C0"),
               ("watermark_ber", "Watermark BER", "C1")]

    for lam, color in zip(active, colors):
        rep_data = [load_pretrain_metrics(r["run_id"]) for r in lam_runs[lam]]
        min_e = max(d["epoch"][0] for d in rep_data)
        max_e = min(d["epoch"][-1] for d in rep_data)
        common = [int(e) for e in range(int(min_e), int(max_e) + 1)]
        aligned = interpolate(common, rep_data)

        for ax, (mname, label, _) in zip(axes, metrics):
            mat = np.array([d[mname] for d in aligned])
            mean = np.mean(mat, axis=0)
            std = np.std(mat, axis=0)
            ax.plot(common, mean, "-", color=color, linewidth=2, label=f"λ={lam}")
            ax.fill_between(common, mean - std, mean + std, color=color, alpha=0.15)

    for ax, (mname, label, mcolor) in zip(axes, metrics):
        ax.set_xlabel("Pretrain Epoch", fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(f"{label} during pretraining\n(5000 samples, {ep} epochs)", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    axes[0].set_yscale("log")
    axes[1].set_yscale("log")

    fig.suptitle(f"Pretraining Dynamics — watermark influence vs λ ({ep} pretrain epochs)", fontsize=14, y=1.03)
    fig.tight_layout()
    path = config_dir / "pretrain_evolution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Print table: final epoch values per λ
    print("\n  === PRETRAIN FINAL EPOCH (mean over reps) ===")
    print(f"  {'λ':<8} {'N':>3} {'wm_loss':>10} {'task_loss':>10} {'BER':>7}")
    print(f"  {'-'*8} {'-'*3} {'-'*10} {'-'*10} {'-'*7}")
    for lam in active:
        rep_data = [load_pretrain_metrics(r["run_id"]) for r in lam_runs[lam]]
        n = len(rep_data)
        wml = np.mean([d["watermark_loss"][-1] for d in rep_data])
        tl = np.mean([d["task_loss"][-1] for d in rep_data])
        ber = np.mean([d["watermark_ber"][-1] for d in rep_data])
        print(f"  {lam:<8} {n:>3} {wml:>10.4f} {tl:>10.4f} {ber:>7.4f}")


if __name__ == "__main__":
    main()
