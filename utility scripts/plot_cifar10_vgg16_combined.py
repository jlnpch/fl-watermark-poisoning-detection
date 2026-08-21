"""CIFAR-10 VGG-16-BN combined baseline analysis: no-WM + WM side by side."""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path("/home2/julian/test/fl-watermark-poisoning-detection/results")
OUT_DIR = RESULTS_DIR / "plots" / "cifar10_vgg16"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_registry():
    with open(RESULTS_DIR / "run_registry.jsonl") as f:
        return [json.loads(line.strip()) for line in f]


def find_runs(runs, wm_lambda, min_rounds=10):
    matched = []
    seen = set()
    for r in runs:
        rid = r["run_id"]
        if rid in seen:
            continue
        if (r.get("model-name") == "vgg16"
                and r.get("dataset-name") == "uoft-cs/cifar10"
                and abs(r.get("watermark-lambda", 0) - wm_lambda) < 1e-9
                and r.get("attacker-type", "none") == "none"):
            srv = RESULTS_DIR / f"run_{rid}_server.csv"
            if srv.exists():
                with open(srv) as sf:
                    n = sum(1 for _ in csv.DictReader(sf))
                if n >= min_rounds:
                    seen.add(rid)
                    matched.append(r)
    return matched


def load_server(run_id):
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


def load_train(run_id):
    path = RESULTS_DIR / f"run_{run_id}_train.csv"
    if not path.exists():
        return None
    by_round = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            by_round[int(row["round"])].append(float(row["watermark_ber"]))
    rounds = sorted(by_round)
    means = [np.mean(by_round[r]) for r in rounds]
    stds = [np.std(by_round[r]) for r in rounds]
    return rounds, means, stds


def load_pretrain(run_id):
    path = RESULTS_DIR / f"run_{run_id}_pretrain.csv"
    if not path.exists():
        return None
    epochs, tl, wl, wb = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            tl.append(float(row["task_loss"]))
            wl.append(float(row["watermark_loss"]))
            wb.append(float(row["watermark_ber"]))
    return epochs, tl, wl, wb


def interpolate_to_common(common, data_list):
    mat = []
    for rnds, vals in data_list:
        rmap = dict(zip(rnds, vals))
        mat.append([rmap.get(r, np.nan) for r in common])
    return np.array(mat)


def main():
    runs = load_registry()

    nowm_runs = find_runs(runs, wm_lambda=0)
    wm_runs = find_runs(runs, wm_lambda=0.01)

    print(f"Found {len(nowm_runs)} no-WM and {len(wm_runs)} WM runs")

    # Load data
    nowm_server = []
    wm_server = []
    wm_ber = []
    nowm_pretrain = []
    wm_pretrain = []

    for r in nowm_runs:
        s = load_server(r["run_id"])
        if s:
            nowm_server.append(s)
        p = load_pretrain(r["run_id"])
        if p:
            nowm_pretrain.append(p)

    for r in wm_runs:
        s = load_server(r["run_id"])
        if s:
            wm_server.append(s)
        t = load_train(r["run_id"])
        if t:
            wm_ber.append(t)
        p = load_pretrain(r["run_id"])
        if p:
            wm_pretrain.append(p)

    # === Plot 1: Accuracy comparison ===
    fig, ax = plt.subplots(figsize=(10, 6))
    common_nowm = list(range(1, max(r[0][-1] for r in nowm_server) + 1))
    common_wm = list(range(1, max(r[0][-1] for r in wm_server) + 1))
    nowm_acc_mat = interpolate_to_common(common_nowm, [(d[0], d[1]) for d in nowm_server])
    wm_acc_mat = interpolate_to_common(common_wm, [(d[0], d[1]) for d in wm_server])

    nowm_acc_mean = np.nanmean(nowm_acc_mat, axis=0)
    nowm_acc_std = np.nanstd(nowm_acc_mat, axis=0)
    wm_acc_mean = np.nanmean(wm_acc_mat, axis=0)
    wm_acc_std = np.nanstd(wm_acc_mat, axis=0)

    ax.plot(common_nowm, nowm_acc_mean, "-", color="C0", linewidth=2.5, label="No-WM mean")
    ax.fill_between(common_nowm, nowm_acc_mean - nowm_acc_std, nowm_acc_mean + nowm_acc_std,
                    color="C0", alpha=0.2)
    ax.plot(common_wm, wm_acc_mean, "-", color="C1", linewidth=2.5, label="WM mean")
    ax.fill_between(common_wm, wm_acc_mean - wm_acc_std, wm_acc_mean + wm_acc_std,
                    color="C1", alpha=0.2)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Server Accuracy (%)", fontsize=12)
    ax.set_title("VGG-16-BN on CIFAR-10 — Accuracy\n(3 reps each, no-WM vs WM, no attack)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: accuracy_comparison.png")

    # === Plot 2: Loss comparison ===
    fig, ax = plt.subplots(figsize=(10, 6))
    nowm_loss_mat = interpolate_to_common(common_nowm, [(d[0], d[2]) for d in nowm_server])
    wm_loss_mat = interpolate_to_common(common_wm, [(d[0], d[2]) for d in wm_server])

    nowm_loss_mean = np.nanmean(nowm_loss_mat, axis=0)
    nowm_loss_std = np.nanstd(nowm_loss_mat, axis=0)
    wm_loss_mean = np.nanmean(wm_loss_mat, axis=0)
    wm_loss_std = np.nanstd(wm_loss_mat, axis=0)

    ax.plot(common_nowm, nowm_loss_mean, "-", color="C0", linewidth=2.5, label="No-WM mean")
    ax.fill_between(common_nowm, nowm_loss_mean - nowm_loss_std, nowm_loss_mean + nowm_loss_std,
                    color="C0", alpha=0.2)
    ax.plot(common_wm, wm_loss_mean, "-", color="C1", linewidth=2.5, label="WM mean")
    ax.fill_between(common_wm, wm_loss_mean - wm_loss_std, wm_loss_mean + wm_loss_std,
                    color="C1", alpha=0.2)

    ax.set_xlabel("FL Round", fontsize=12)
    ax.set_ylabel("Server Loss", fontsize=12)
    ax.set_title("VGG-16-BN on CIFAR-10 — Loss\n(3 reps each, no-WM vs WM, no attack)", fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: loss_comparison.png")

    # === Plot 3: BER (WM only) with per-client variance ===
    if wm_ber:
        fig, ax = plt.subplots(figsize=(10, 6))
        all_rounds = sorted(set(r for d in wm_ber for r in d[0]))
        mean_mat, std_mat = [], []
        for rnds, bers, stds in wm_ber:
            rm = dict(zip(rnds, bers))
            rs = dict(zip(rnds, stds))
            mean_mat.append([rm.get(r, np.nan) for r in all_rounds])
            std_mat.append([rs.get(r, np.nan) for r in all_rounds])
        mean_mat = np.array(mean_mat)
        std_mat = np.array(std_mat)

        agg_mean = np.nanmean(mean_mat, axis=0)
        agg_std = np.nanmean(std_mat, axis=0)
        rep_std = np.nanstd(mean_mat, axis=0)

        ax.plot(all_rounds, agg_mean, "-", color="C2", linewidth=2.5, label="Mean BER")
        ax.fill_between(all_rounds, agg_mean - agg_std, agg_mean + agg_std,
                        color="C2", alpha=0.2, label="Mean ± client std")

        ax.set_xlabel("FL Round", fontsize=12)
        ax.set_ylabel("Watermark BER", fontsize=12)
        ax.set_title("VGG-16-BN on CIFAR-10 — BER (classifier.6.weight)\n(3 WM reps, 128-bit, λ=0.01)", fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 0.55)
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "ber_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: ber_comparison.png")

    # === Plot 4: Pretrain (both) ===
    if nowm_pretrain or wm_pretrain:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for data, color, prefix in [(nowm_pretrain, "C0", "No-WM"), (wm_pretrain, "C1", "WM")]:
            if not data:
                continue
            common_ep = data[0][0]
            ep_map_task, ep_map_ber = {}, {}
            for epochs, tl, wl, wb in data:
                for i, e in enumerate(epochs):
                    ep_map_task.setdefault(e, []).append(tl[i])
                    ep_map_ber.setdefault(e, []).append(wb[i])

            tl_mean = [np.mean(ep_map_task[e]) for e in common_ep]
            tl_std = [np.std(ep_map_task[e]) for e in common_ep]
            wb_mean = [np.mean(ep_map_ber[e]) for e in common_ep]
            wb_std = [np.std(ep_map_ber[e]) for e in common_ep]
            axes[0].plot(common_ep, tl_mean, "o-", color=color, linewidth=2, markersize=3, label=prefix)
            axes[0].fill_between(common_ep, np.array(tl_mean) - np.array(tl_std),
                                np.array(tl_mean) + np.array(tl_std), color=color, alpha=0.15)
            axes[2].plot(common_ep, wb_mean, "o-", color=color, linewidth=2, markersize=3, label=prefix)
            axes[2].fill_between(common_ep, np.array(wb_mean) - np.array(wb_std),
                                np.array(wb_mean) + np.array(wb_std), color=color, alpha=0.15)

        if wm_pretrain:
            wm_wl_map = {}
            for epochs, tl, wl, wb in wm_pretrain:
                for i, e in enumerate(epochs):
                    wm_wl_map.setdefault(e, []).append(wl[i])
            wm_ep = sorted(wm_wl_map.keys())
            wl_mean = [np.mean(wm_wl_map[e]) for e in wm_ep]
            wl_std = [np.std(wm_wl_map[e]) for e in wm_ep]
            axes[1].plot(wm_ep, wl_mean, "o-", color="C1", linewidth=2, markersize=3, label="WM mean")
            axes[1].fill_between(wm_ep, np.array(wl_mean) - np.array(wl_std),
                                np.array(wl_mean) + np.array(wl_std), color="C1", alpha=0.15)

        axes[0].set_yscale("log")
        axes[0].set_title("Task Loss")
        axes[1].set_yscale("log")
        axes[1].set_title("Watermark Loss (WM only)")
        axes[2].set_title("Watermark BER")
        for ax in axes:
            ax.set_xlabel("Pretrain Epoch")
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(fontsize=10)
        fig.suptitle("VGG-16-BN CIFAR-10 Pretrain: No-WM vs WM", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(OUT_DIR / "pretrain_comparison.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: pretrain_comparison.png")

    # === Summary ===
    print(f"\n=== CIFAR-10 VGG-16-BN Summary ===")
    print(f"\nNo-Watermark ({len(nowm_server)} reps):")
    final_accs = [d[1][-1] for d in nowm_server]
    final_losses = [d[2][-1] for d in nowm_server]
    for i, (rnds, accs, losses) in enumerate(nowm_server):
        best_idx = accs.index(max(accs))
        print(f"  Rep {i+1}: {rnds[-1]} rounds (early stop), best acc={max(accs):.2f}% at round {rnds[best_idx]}, final loss={losses[-1]:.4f}")
    print(f"  Mean final acc: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%")

    print(f"\nWatermark ({len(wm_server)} reps):")
    final_accs_wm = [d[1][-1] for d in wm_server]
    for i, (rnds, accs, losses) in enumerate(wm_server):
        best_idx = accs.index(max(accs))
        print(f"  Rep {i+1}: {rnds[-1]} rounds (early stop), best acc={max(accs):.2f}% at round {rnds[best_idx]}, final loss={losses[-1]:.4f}")
    print(f"  Mean final acc: {np.mean(final_accs_wm):.2f}% ± {np.std(final_accs_wm):.2f}%")

    if wm_ber:
        final_bers = [d[1][-1] for d in wm_ber]
        print(f"\n  Mean final BER: {np.mean(final_bers):.4f} ± {np.std(final_bers):.4f}")


if __name__ == "__main__":
    main()
