#!/usr/bin/env python3
"""Plot CIFAR-100 VGG-16 (no-BN) baseline runs — accuracy & BER evolution.

Two conditions, 3 reps each:
  - No-watermark (λ=0):   183531, 195522, 211532
  - Watermark (λ=0.01):   223536, 235521, 011534
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

# Run IDs from run_registry
BASELINE_RUNS = ["20260818_183531", "20260818_195522", "20260818_211532"]
WATERMARK_RUNS = ["20260818_223536", "20260818_235521", "20260819_011534"]


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

    # No-watermark
    baseline_data = [load_server_data(rid) for rid in BASELINE_RUNS]
    min_r = max(d[0][0] for d in baseline_data)
    max_r = min(d[0][-1] for d in baseline_data)
    cr = list(range(min_r, max_r + 1))
    mat = interpolate(cr, [(d[0], d[1]) for d in baseline_data])
    mean_bl = np.nanmean(mat, axis=0)
    std_bl = np.nanstd(mat, axis=0)
    ax.plot(cr, mean_bl, "-", color="C0", linewidth=2, label="No watermark (λ=0)")
    ax.fill_between(cr, mean_bl - std_bl, mean_bl + std_bl, color="C0", alpha=0.15)

    # Watermark
    wm_data = [load_server_data(rid) for rid in WATERMARK_RUNS]
    min_r = max(d[0][0] for d in wm_data)
    max_r = min(d[0][-1] for d in wm_data)
    cr2 = list(range(min_r, max_r + 1))
    mat2 = interpolate(cr2, [(d[0], d[1]) for d in wm_data])
    mean_wm = np.nanmean(mat2, axis=0)
    std_wm = np.nanstd(mat2, axis=0)
    ax.plot(cr2, mean_wm, "-", color="C1", linewidth=2, label="Watermark (λ=0.01)")
    ax.fill_between(cr2, mean_wm - std_wm, mean_wm + std_wm, color="C1", alpha=0.15)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("CIFAR-100 VGG-16 — Accuracy Evolution\n(3 reps, 50 pretrain epochs, IID, 10 clients)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "accuracy_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'accuracy_evolution.png'}")


def plot_loss():
    fig, ax = plt.subplots(figsize=(10, 6))

    baseline_data = [load_server_data(rid) for rid in BASELINE_RUNS]
    min_r = max(d[0][0] for d in baseline_data)
    max_r = min(d[0][-1] for d in baseline_data)
    cr = list(range(min_r, max_r + 1))
    mat = interpolate(cr, [(d[0], d[2]) for d in baseline_data])
    mean_bl = np.nanmean(mat, axis=0)
    std_bl = np.nanstd(mat, axis=0)
    ax.plot(cr, mean_bl, "-", color="C0", linewidth=2, label="No watermark (λ=0)")
    ax.fill_between(cr, mean_bl - std_bl, mean_bl + std_bl, color="C0", alpha=0.15)

    wm_data = [load_server_data(rid) for rid in WATERMARK_RUNS]
    min_r = max(d[0][0] for d in wm_data)
    max_r = min(d[0][-1] for d in wm_data)
    cr2 = list(range(min_r, max_r + 1))
    mat2 = interpolate(cr2, [(d[0], d[2]) for d in wm_data])
    mean_wm = np.nanmean(mat2, axis=0)
    std_wm = np.nanstd(mat2, axis=0)
    ax.plot(cr2, mean_wm, "-", color="C1", linewidth=2, label="Watermark (λ=0.01)")
    ax.fill_between(cr2, mean_wm - std_wm, mean_wm + std_wm, color="C1", alpha=0.15)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("CIFAR-100 VGG-16 — Loss Evolution\n(3 reps, 50 pretrain epochs, IID, 10 clients)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "loss_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {PLOT_DIR / 'loss_evolution.png'}")


def plot_ber():
    fig, ax = plt.subplots(figsize=(10, 6))

    # Watermark runs BER
    wm_data = [load_train_ber(rid) for rid in WATERMARK_RUNS]
    min_r = max(d[0][0] for d in wm_data)
    max_r = min(d[0][-1] for d in wm_data)
    cr = list(range(min_r, max_r + 1))
    mat = interpolate(cr, wm_data)
    mean_wm = np.nanmean(mat, axis=0)
    std_wm = np.nanstd(mat, axis=0)
    ax.plot(cr, mean_wm, "-", color="C1", linewidth=2, label="Watermark (λ=0.01)")
    ax.fill_between(cr, mean_wm - std_wm, mean_wm + std_wm, color="C1", alpha=0.15)

    # Baseline runs BER
    bl_data = [load_train_ber(rid) for rid in BASELINE_RUNS]
    min_r2 = max(d[0][0] for d in bl_data)
    max_r2 = min(d[0][-1] for d in bl_data)
    cr2 = list(range(min_r2, max_r2 + 1))
    mat2 = interpolate(cr2, bl_data)
    mean_bl = np.nanmean(mat2, axis=0)
    std_bl = np.nanstd(mat2, axis=0)
    ax.plot(cr2, mean_bl, "--", color="C0", linewidth=2, label="No watermark (λ=0)")
    ax.fill_between(cr2, mean_bl - std_bl, mean_bl + std_bl, color="C0", alpha=0.15)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Watermark BER", fontsize=12)
    ax.set_title("CIFAR-100 VGG-16 — BER Evolution\n(3 reps, 50 pretrain epochs, IID, 10 clients)", fontsize=13)
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
