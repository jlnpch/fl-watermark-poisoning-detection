#!/usr/bin/env python3
"""Plot baseline sweep: 2 images with subplots.

- baseline_ber.png: 4 subplots (one per server size), 3 lines each (epochs 10/30/50)
- baseline_acc.png: same layout for accuracy
"""

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
PLOTS_DIR = RESULTS_DIR / "plots" / "baseline"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

SERVER_SIZES = [500, 2500, 5000, 10000]
PRETRAIN_EPOCHS = [10, 30, 50]
EPOCH_COLORS = {10: "C0", 30: "C1", 50: "C2"}


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


def find_runs(runs, server_samples, pretrain_epochs, watermark_lambda=0.01):
    return [r for r in runs
            if r.get("server-private-samples") == server_samples
            and r.get("pretrain-epochs") == pretrain_epochs
            and r.get("attacker-type", "none") == "none"
            and r.get("attacker-fraction", 0) == 0
            and abs(r.get("watermark-lambda", 0.01) - watermark_lambda) < 1e-9]


def load_train_ber(run_id):
    path = RESULTS_DIR / f"run_{run_id}_train.csv"
    if not path.exists():
        return None
    by_round = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_round[int(row["round"])].append(float(row["watermark_ber"]))
    rounds = sorted(by_round)
    return rounds, [np.mean(by_round[r]) for r in rounds]


def load_server_acc(run_id):
    path = RESULTS_DIR / f"run_{run_id}_server.csv"
    if not path.exists():
        return None
    rounds, accs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            accs.append(float(row["server_acc"]) * 100)
    return rounds, accs


def interpolate(common_rounds, rep_data):
    result = []
    for rounds, values in rep_data:
        rmap = dict(zip(rounds, values))
        result.append([rmap.get(r, np.nan) for r in common_rounds])
    return np.array(result)


def main():
    runs = load_registry()
    print(f"Found {len(runs)} runs in registry")

    # Collect data: (samples, epochs) -> list of (rounds, values)
    all_ber = defaultdict(list)
    all_acc = defaultdict(list)

    for samples in SERVER_SIZES:
        for epochs in PRETRAIN_EPOCHS:
            matches = find_runs(runs, samples, epochs)
            print(f"  samples={samples} epochs={epochs}: {len(matches)} runs")
            for m in matches:
                rid = m["run_id"]
                d = load_train_ber(rid)
                if d:
                    all_ber[(samples, epochs)].append(d)
                d = load_server_acc(rid)
                if d:
                    all_acc[(samples, epochs)].append(d)

    # --- BER figure ---
    fig_ber, axes_ber = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    for ax, samples in zip(axes_ber, SERVER_SIZES):
        for epochs in PRETRAIN_EPOCHS:
            key = (samples, epochs)
            if key not in all_ber or not all_ber[key]:
                continue
            rep_data = all_ber[key]
            min_r = max(r[0][0] for r in rep_data)
            max_r = min(r[0][-1] for r in rep_data)
            cr = list(range(min_r, max_r + 1))
            mat = interpolate(cr, rep_data)
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            ax.plot(cr, mean, "-", color=EPOCH_COLORS[epochs], linewidth=2,
                    label=f"{epochs} epochs")
            ax.fill_between(cr, mean - std, mean + std, color=EPOCH_COLORS[epochs], alpha=0.15)

        ax.set_xlabel("FL Round", fontsize=11)
        ax.set_title(f"Server size = {samples}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 0.75)
        ax.legend(fontsize=9)

    axes_ber[0].set_ylabel("Watermark BER", fontsize=12)
    fig_ber.suptitle("Baseline BER — 3 pretrain epoch settings × 4 server sizes", fontsize=14, y=1.02)
    fig_ber.tight_layout()
    fig_ber.savefig(PLOTS_DIR / "baseline_ber.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ber)
    print(f"  Saved: {PLOTS_DIR / 'baseline_ber.png'}")

    # --- Accuracy figure ---
    fig_acc, axes_acc = plt.subplots(1, 4, figsize=(22, 5), sharey=True)
    for ax, samples in zip(axes_acc, SERVER_SIZES):
        for epochs in PRETRAIN_EPOCHS:
            key = (samples, epochs)
            if key not in all_acc or not all_acc[key]:
                continue
            rep_data = all_acc[key]
            min_r = max(r[0][0] for r in rep_data)
            max_r = min(r[0][-1] for r in rep_data)
            cr = list(range(min_r, max_r + 1))
            mat = interpolate(cr, rep_data)
            mean = np.nanmean(mat, axis=0)
            std = np.nanstd(mat, axis=0)
            ax.plot(cr, mean, "-", color=EPOCH_COLORS[epochs], linewidth=2,
                    label=f"{epochs} epochs")
            ax.fill_between(cr, mean - std, mean + std, color=EPOCH_COLORS[epochs], alpha=0.15)

        ax.set_xlabel("FL Round", fontsize=11)
        ax.set_title(f"Server size = {samples}", fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=9)

    axes_acc[0].set_ylabel("Server Accuracy (%)", fontsize=12)
    fig_acc.suptitle("Baseline Accuracy — 3 pretrain epoch settings × 4 server sizes", fontsize=14, y=1.02)
    fig_acc.tight_layout()
    fig_acc.savefig(PLOTS_DIR / "baseline_acc.png", dpi=150, bbox_inches="tight")
    plt.close(fig_acc)
    print(f"  Saved: {PLOTS_DIR / 'baseline_acc.png'}")


def plot_lambda_scan():
    """Plot clean baseline BER and accuracy as function of λ (5000/30 config)."""
    runs = load_registry()
    lambdas = [0.005, 0.01, 0.05, 0.1, 0.25, 0.75]

    lam_data = {}
    for lam in lambdas:
        matches = [r for r in runs
                   if r.get("attacker-type", "none") == "none"
                   and r.get("attacker-fraction", 0) == 0
                   and r.get("server-private-samples") == 5000
                   and r.get("pretrain-epochs") == 30
                   and abs(r.get("watermark-lambda", 0) - lam) < 1e-9]
        bers, accs = [], []
        for m in matches:
            rid = m["run_id"]
            train = load_train_ber(rid)
            server = load_server_acc(rid)
            if train:
                bers.append(np.mean(train[1]))
            if server:
                accs.append(np.mean(server[1]))
            if len(bers) >= 3:
                break
        lam_data[lam] = (bers, accs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # BER subplot
    ax = axes[0]
    x = [str(lam) for lam in lambdas]
    for i, lam in enumerate(lambdas):
        bers, _ = lam_data[lam]
        for b in bers:
            ax.plot(i, b, "o", color="C0", markersize=4, alpha=0.5, zorder=3)
    means = [np.mean(lam_data[lam][0]) if lam_data[lam][0] else 0 for lam in lambdas]
    stds = [np.std(lam_data[lam][0]) if lam_data[lam][0] else 0 for lam in lambdas]
    ax.errorbar(range(len(x)), means, yerr=stds, fmt="o-", capsize=5, capthick=1.5, linewidth=2, markersize=8, color="C0")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, fontsize=10)
    ax.set_xlabel("Watermark Strength (λ)")
    ax.set_ylabel("Watermark BER")
    ax.set_title("Clean Baseline BER vs λ\n(5000 server samples, 30 pretrain epochs)")
    ax.grid(True, alpha=0.3)

    # Accuracy subplot
    ax = axes[1]
    for i, lam in enumerate(lambdas):
        _, accs = lam_data[lam]
        for a in accs:
            ax.plot(i, a, "s", color="C2", markersize=4, alpha=0.5, zorder=3)
    means = [np.mean(lam_data[lam][1]) if lam_data[lam][1] else 0 for lam in lambdas]
    stds = [np.std(lam_data[lam][1]) if lam_data[lam][1] else 0 for lam in lambdas]
    ax.errorbar(range(len(x)), means, yerr=stds, fmt="s-", capsize=5, capthick=1.5, linewidth=2, markersize=8, color="C2")
    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x, fontsize=10)
    ax.set_xlabel("Watermark Strength (λ)")
    ax.set_ylabel("Server Accuracy (%)")
    ax.set_title("Clean Baseline Accuracy vs λ\n(5000 server samples, 30 pretrain epochs)")
    ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Watermark Strength on Clean Baseline", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "baseline_lambda_scan.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {PLOTS_DIR / 'baseline_lambda_scan.png'}")

    # Print table with sample sizes
    print("\n  === LAMBDA SCAN (clean baseline, 5000/30) ===")
    print(f"  {'λ':<8} {'N':>3} {'Mean BER':<10} {'Std BER':<10} {'Mean Acc':<10} {'Std Acc':<10}")
    print(f"  {'-'*8} {'-'*3} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for lam in lambdas:
        bers, accs = lam_data[lam]
        n = len(bers)
        mb = np.mean(bers) if bers else 0
        sb = np.std(bers) if bers else 0
        ma = np.mean(accs) if accs else 0
        sa = np.std(accs) if accs else 0
        print(f"  {lam:<8} {n:>3} {mb:<10.4f} {sb:<10.4f} {ma:<10.2f} {sa:<10.2f}")


if __name__ == "__main__":
    main()
    plot_lambda_scan()
