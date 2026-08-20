#!/usr/bin/env python3
"""Plot VGG-16 CIFAR-10 baseline runs: accuracy, loss, and BER evolution."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
OUT_DIR = RESULTS_DIR / "plots" / "cifar10_vgg16"
OUT_DIR.mkdir(parents=True, exist_ok=True)

REGISTRY = RESULTS_DIR / "run_registry.jsonl"


def load_registry():
    runs = []
    with open(REGISTRY) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs


def find_vgg16_cifar10_runs(runs, watermark_lambda=0):
    seen = set()
    matched = []
    for r in runs:
        rid = r["run_id"]
        if rid in seen:
            continue
        if (r.get("model-name") == "vgg16"
                and r.get("dataset-name") == "uoft-cs/cifar10"
                and abs(r.get("watermark-lambda", 0) - watermark_lambda) < 1e-9
                and r.get("attacker-type", "none") == "none"):
            srv = RESULTS_DIR / f"run_{rid}_server.csv"
            if srv.exists():
                seen.add(rid)
                matched.append(r)
    return matched


def load_server_csv(run_id):
    path = RESULTS_DIR / f"run_{run_id}_server.csv"
    if not path.exists():
        return None
    rounds, accs, losses = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            accs.append(float(row["server_acc"]) * 100)
            losses.append(float(row["server_loss"]))
    return rounds, accs, losses


def load_train_csv(run_id):
    path = RESULTS_DIR / f"run_{run_id}_train.csv"
    if not path.exists():
        return None
    by_round = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_round[int(row["round"])].append(float(row["watermark_ber"]))
    rounds = sorted(by_round)
    bers = [np.mean(by_round[r]) for r in rounds]
    return rounds, bers


def load_pretrain_csv(run_id):
    path = RESULTS_DIR / f"run_{run_id}_pretrain.csv"
    if not path.exists():
        return None
    epochs, task_losses, wm_losses, wm_bers = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            task_losses.append(float(row["task_loss"]))
            wm_losses.append(float(row["watermark_loss"]))
            wm_bers.append(float(row["watermark_ber"]))
    return epochs, task_losses, wm_losses, wm_bers


def interpolate(common_rounds, rep_data):
    result = []
    for rounds, values in rep_data:
        rmap = dict(zip(rounds, values))
        result.append([rmap.get(r, np.nan) for r in common_rounds])
    return np.array(result)


def main():
    runs = load_registry()

    baseline_runs = find_vgg16_cifar10_runs(runs, watermark_lambda=0)
    print(f"Found {len(baseline_runs)} VGG-16 CIFAR-10 baseline runs (no watermark)")

    server_data = []
    train_ber_data = []
    pretrain_data = []

    for r in baseline_runs:
        rid = r["run_id"]
        s = load_server_csv(rid)
        if s:
            server_data.append(s)
        t = load_train_csv(rid)
        if t:
            train_ber_data.append(t)
        p = load_pretrain_csv(rid)
        if p:
            pretrain_data.append(p)

    if not server_data:
        print("No server data found!")
        return

    # --- Accuracy ---
    fig, ax = plt.subplots(figsize=(10, 6))
    min_r = max(d[0][0] for d in server_data)
    max_r = min(d[0][-1] for d in server_data)
    common_rounds = list(range(min_r, max_r + 1))

    acc_matrix = interpolate(common_rounds, [(d[0], d[1]) for d in server_data])
    loss_matrix = interpolate(common_rounds, [(d[0], d[2]) for d in server_data])

    acc_mean = np.nanmean(acc_matrix, axis=0)
    acc_std = np.nanstd(acc_matrix, axis=0)
    loss_mean = np.nanmean(loss_matrix, axis=0)
    loss_std = np.nanstd(loss_matrix, axis=0)

    for i, (rnds, accs, _) in enumerate(server_data):
        ax.plot(rnds, accs, "--", alpha=0.3, linewidth=1,
                label=f"Rep {i+1}" if i < 5 else None)

    ax.plot(common_rounds, acc_mean, "-", color="C0", linewidth=2.5, label="Mean")
    ax.fill_between(common_rounds, acc_mean - acc_std, acc_mean + acc_std,
                    color="C0", alpha=0.15)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Server Accuracy (%)", fontsize=12)
    ax.set_title(
        f"VGG-16-BN on CIFAR-10 -- Baseline Accuracy\n"
        f"({len(server_data)} reps, no watermark, no attack)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'accuracy_evolution.png'}")

    # --- Loss ---
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (rnds, _, loss) in enumerate(server_data):
        ax.plot(rnds, loss, "--", alpha=0.3, linewidth=1,
                label=f"Rep {i+1}" if i < 5 else None)

    ax.plot(common_rounds, loss_mean, "-", color="C1", linewidth=2.5, label="Mean")
    ax.fill_between(common_rounds, loss_mean - loss_std, loss_mean + loss_std,
                    color="C1", alpha=0.15)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Server Loss", fontsize=12)
    ax.set_title(
        f"VGG-16-BN on CIFAR-10 -- Baseline Loss\n"
        f"({len(server_data)} reps, no watermark, no attack)",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {OUT_DIR / 'loss_evolution.png'}")

    # --- BER ---
    if train_ber_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        min_r = max(d[0][0] for d in train_ber_data)
        max_r = min(d[0][-1] for d in train_ber_data)
        common_rounds_ber = list(range(min_r, max_r + 1))
        ber_matrix = interpolate(common_rounds_ber, train_ber_data)
        ber_mean = np.nanmean(ber_matrix, axis=0)
        ber_std = np.nanstd(ber_matrix, axis=0)

        for i, (rnds, bers) in enumerate(train_ber_data):
            ax.plot(rnds, bers, "--", alpha=0.3, linewidth=1,
                    label=f"Rep {i+1}" if i < 5 else None)

        ax.plot(common_rounds_ber, ber_mean, "-", color="C2", linewidth=2.5,
                label="Mean")
        ax.fill_between(common_rounds_ber, ber_mean - ber_std, ber_mean + ber_std,
                        color="C2", alpha=0.15)

        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("Watermark BER", fontsize=12)
        ax.set_title(
            f"VGG-16-BN on CIFAR-10 -- Baseline BER\n"
            f"({len(train_ber_data)} reps, no watermark, no attack)",
            fontsize=13,
        )
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "ber_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUT_DIR / 'ber_evolution.png'}")

    # --- Pretrain ---
    if pretrain_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        epochs_p = pretrain_data[0][0]
        for epochs, tl, wl, wb in pretrain_data:
            axes[0].plot(epochs, tl, "o-", alpha=0.5, markersize=3)
            axes[1].plot(epochs, wl, "o-", alpha=0.5, markersize=3)
            axes[2].plot(epochs, wb, "o-", alpha=0.5, markersize=3)

        task_means = np.mean([d[1] for d in pretrain_data], axis=0)
        wm_means = np.mean([d[2] for d in pretrain_data], axis=0)
        ber_means = np.mean([d[3] for d in pretrain_data], axis=0)
        axes[0].plot(epochs_p, task_means, "o-", color="C0", linewidth=2, label="Mean")
        axes[1].plot(epochs_p, wm_means, "o-", color="C1", linewidth=2, label="Mean")
        axes[2].plot(epochs_p, ber_means, "o-", color="C2", linewidth=2, label="Mean")

        axes[0].set_title("Task Loss")
        axes[1].set_title("Watermark Loss")
        axes[2].set_title("Watermark BER")
        for ax in axes:
            ax.set_xlabel("Pretrain Epoch")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        axes[0].set_ylabel("Loss")

        fig.suptitle(
            f"VGG-16-BN on CIFAR-10 -- Pretrain Phase ({len(pretrain_data)} reps)",
            fontsize=13,
        )
        fig.tight_layout()
        fig.savefig(OUT_DIR / "pretrain_evolution.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {OUT_DIR / 'pretrain_evolution.png'}")

    # --- Summary ---
    print(f"\n=== VGG-16 CIFAR-10 Baseline Summary ===")
    final_accs = [d[1][-1] for d in server_data]
    final_losses = [d[2][-1] for d in server_data]
    print(f"  Reps: {len(server_data)}")
    print(f"  Final accuracy: {np.mean(final_accs):.2f}% +/- {np.std(final_accs):.2f}%")
    print(f"  Final loss: {np.mean(final_losses):.4f} +/- {np.std(final_losses):.4f}")
    if train_ber_data:
        final_bers = [d[1][-1] for d in train_ber_data]
        print(f"  Final BER: {np.mean(final_bers):.4f} +/- {np.std(final_bers):.4f}")


if __name__ == "__main__":
    main()
