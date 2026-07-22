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
PLOTS_DIR = RESULTS_DIR / "plots"
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


def find_runs(runs, server_samples, pretrain_epochs):
    return [r for r in runs
            if r.get("server-private-samples") == server_samples
            and r.get("pretrain-epochs") == pretrain_epochs
            and r.get("attacker-type", "none") == "none"]


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


if __name__ == "__main__":
    main()
