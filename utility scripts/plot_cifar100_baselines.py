#!/usr/bin/env python3
"""Plot CIFAR-100 VGG-16-BN baseline runs — accuracy & BER evolution.

Two conditions, 1 rep each (200 rounds):
  - No-watermark (λ=0):   20260819_135755
  - Watermark (λ=0.01):   20260819_163844
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
PLOT_DIR = RESULTS_DIR / "plots" / "cifar100_vgg16"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Run IDs from run_registry (VGG-16-BN, 200 rounds)
BASELINE_RUNS = ["20260819_135755"]
WATERMARK_RUNS = ["20260819_163844"]


def load_server_data(run_id):
    """Return (rounds, accs, losses)."""
    path = RESULTS_DIR / f"run_{run_id}_server.csv"
    rounds, accs, losses = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            accs.append(float(row["server_acc"]) * 100)
            losses.append(float(row["server_loss"]))
    return rounds, accs, losses


def load_train_ber(run_id):
    """Return (rounds, mean_ber_per_round)."""
    path = RESULTS_DIR / f"run_{run_id}_train.csv"
    by_round = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_round[int(row["round"])].append(float(row["watermark_ber"]))
    rounds = sorted(by_round)
    return rounds, [np.mean(by_round[r]) for r in rounds]


def interpolate(common_rounds, rep_data):
    """rep_data = list of (rounds, values). Returns 2D array (n_reps, n_rounds)."""
    result = []
    for rounds, values in rep_data:
        rmap = dict(zip(rounds, values))
        result.append([rmap.get(r, np.nan) for r in common_rounds])
    return np.array(result)


def plot_accuracy():
    fig, ax = plt.subplots(figsize=(10, 6))

    for runs, color, label in [(BASELINE_RUNS, "C0", "No watermark (λ=0)"),
                                (WATERMARK_RUNS, "C1", "Watermark (λ=0.01)")]:
        data = [load_server_data(rid) for rid in runs]
        min_r = max(d[0][0] for d in data)
        max_r = min(d[0][-1] for d in data)
        cr = list(range(min_r, max_r + 1))
        mat = interpolate(cr, [(d[0], d[1]) for d in data])
        mean = np.nanmean(mat, axis=0)
        ax.plot(cr, mean, "-", color=color, linewidth=2, label=label)
        if len(runs) > 1:
            std = np.nanstd(mat, axis=0)
            ax.fill_between(cr, mean - std, mean + std, color=color, alpha=0.15)
            for i, (rnds, accs, _) in enumerate(data):
                ax.plot(rnds, accs, "--", color=color, alpha=0.3, linewidth=1)
        else:
            ax.plot(data[0][0], data[0][1], "--", color=color, alpha=0.4, linewidth=1, label="_nolegend_")

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("CIFAR-100 VGG-16-BN — Accuracy Evolution\n(1 rep, 200 FL rounds, 50 pretrain epochs)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "accuracy_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'accuracy_evolution.png'}")


def plot_loss():
    fig, ax = plt.subplots(figsize=(10, 6))

    for runs, color, label in [(BASELINE_RUNS, "C0", "No watermark (λ=0)"),
                                (WATERMARK_RUNS, "C1", "Watermark (λ=0.01)")]:
        data = [load_server_data(rid) for rid in runs]
        min_r = max(d[0][0] for d in data)
        max_r = min(d[0][-1] for d in data)
        cr = list(range(min_r, max_r + 1))
        mat = interpolate(cr, [(d[0], d[2]) for d in data])
        mean = np.nanmean(mat, axis=0)
        ax.plot(cr, mean, "-", color=color, linewidth=2, label=label)
        if len(runs) > 1:
            std = np.nanstd(mat, axis=0)
            ax.fill_between(cr, mean - std, mean + std, color=color, alpha=0.15)
            for i, (rnds, _, losses) in enumerate(data):
                ax.plot(rnds, losses, "--", color=color, alpha=0.3, linewidth=1)
        else:
            ax.plot(data[0][0], data[0][2], "--", color=color, alpha=0.4, linewidth=1, label="_nolegend_")

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("CIFAR-100 VGG-16-BN — Loss Evolution\n(1 rep, 200 FL rounds, 50 pretrain epochs)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loss_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'loss_evolution.png'}")


def plot_ber():
    fig, ax = plt.subplots(figsize=(10, 6))

    for runs, color, label in [(WATERMARK_RUNS, "C1", "Watermark (λ=0.01)"),
                                (BASELINE_RUNS, "C0", "No watermark (λ=0)")]:
        data = [load_train_ber(rid) for rid in runs]
        min_r = max(d[0][0] for d in data)
        max_r = min(d[0][-1] for d in data)
        cr = list(range(min_r, max_r + 1))
        mat = interpolate(cr, data)
        mean = np.nanmean(mat, axis=0)
        ax.plot(cr, mean, "-" if color == "C1" else "--", color=color, linewidth=2, label=label)
        if len(runs) > 1:
            std = np.nanstd(mat, axis=0)
            ax.fill_between(cr, mean - std, mean + std, color=color, alpha=0.15)
            for rnds, bers in data:
                ax.plot(rnds, bers, "--", color=color, alpha=0.3, linewidth=1)
        else:
            ax.plot(data[0][0], data[0][1], "--", color=color, alpha=0.4, linewidth=1, label="_nolegend_")

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Watermark BER", fontsize=12)
    ax.set_title("CIFAR-100 VGG-16-BN — BER Evolution\n(1 rep, 200 FL rounds, 50 pretrain epochs)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 0.65)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "ber_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'ber_evolution.png'}")


if __name__ == "__main__":
    plot_accuracy()
    plot_loss()
    plot_ber()
