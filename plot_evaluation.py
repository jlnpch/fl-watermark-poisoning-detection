#!/usr/bin/env python3
"""Plot semi-fragility evaluation results.

Generates:
1. eval_ber_evolution.png — BER per config (honest vs attacker, averaged over reps)
2. eval_defense_summary.png — Exclusion accuracy, d'-score, BER separation per config
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
from scipy import stats

RESULTS_DIR = Path("results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Match the 6 configs from run_evaluation.sh
CONFIGS = [
    {"label": "Baseline\n(no attack, λ=0.01)", "attack": "none", "lam": 0.01},
    {"label": "LabelFlip\n(scale=5, λ=0.01)", "attack": "label_flip", "lam": 0.01},
    {"label": "SignFlip\n(scale=5, λ=0.005)", "attack": "sign_flip", "lam": 0.005},
    {"label": "SignFlip\n(scale=5, λ=0.01)", "attack": "sign_flip", "lam": 0.01},
    {"label": "SignFlip\n(scale=5, λ=0.05)", "attack": "sign_flip", "lam": 0.05},
    {"label": "SignFlip\n(scale=5, λ=0.10)", "attack": "sign_flip", "lam": 0.10},
]


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


def find_matching_runs(runs, attack_type, lam):
    """Find runs matching attack type and lambda."""
    return [r for r in runs
            if r.get("attacker-type") == attack_type
            and abs(r.get("watermark-lambda", 0) - lam) < 1e-9
            and r.get("server-private-samples") == 5000
            and r.get("pretrain-epochs") == 30]


def load_train_data(run_id):
    """Load per-round per-client train data. Returns dict: round -> list of (pid, ber, is_attacker, excluded)."""
    path = RESULTS_DIR / f"run_{run_id}_train.csv"
    if not path.exists():
        return None
    data = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            rnd = int(row["round"])
            pid = int(row["partition_id"])
            ber = float(row["watermark_ber"])
            is_att = int(row["is_attacker"])
            excluded = int(row["excluded"])
            data[rnd].append((pid, ber, is_att, excluded))
    return data


def load_server_data(run_id):
    """Load server metrics (accuracy, loss per round)."""
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


def compute_dprime(honest_bers, attacker_bers):
    """Compute d'-score: sensitivity index for BER separation."""
    honest = np.array(honest_bers)
    attacker = np.array(attacker_bers)
    if len(honest) < 2 or len(attacker) < 2:
        return np.nan
    mean_h = np.mean(honest)
    mean_a = np.mean(attacker)
    std_pool = np.sqrt((np.var(honest, ddof=1) + np.var(attacker, ddof=1)) / 2)
    if std_pool < 1e-12:
        return np.nan
    return (mean_h - mean_a) / std_pool


def compute_auc(honest_bers, attacker_bers):
    """Compute ROC-AUC using honest BER (negative class) vs attacker BER (positive class)."""
    honest = np.array(honest_bers)
    attacker = np.array(attacker_bers)
    if len(honest) < 2 or len(attacker) < 2:
        return np.nan
    # Stack: 0 = honest, 1 = attacker
    scores = np.concatenate([honest, attacker])
    labels = np.concatenate([np.zeros(len(honest)), np.ones(len(attacker))])
    # Use scipy's rank-based AUC
    n1 = len(attacker)
    n2 = len(honest)
    rank = stats.rankdata(scores)
    rank_sum = np.sum(rank[len(honest):])
    auc = (rank_sum - n1 * (n1 + 1) / 2) / (n1 * n2)
    return auc


def compute_exclusion_accuracy(train_data):
    """Compute TP, FP, TN, FN for the BER-based defense."""
    tp = fp = tn = fn = 0
    for rnd, clients in train_data.items():
        for pid, ber, is_att, excluded in clients:
            if is_att and excluded:
                tp += 1
            elif not is_att and excluded:
                fp += 1
            elif not is_att and not excluded:
                tn += 1
            elif is_att and not excluded:
                fn += 1
    total = tp + fp + tn + fn
    acc = (tp + tn) / total if total > 0 else 0
    return tp, fp, tn, fn, acc


def main():
    runs = load_registry()
    print(f"Found {len(runs)} runs in registry")

    n_configs = len(CONFIGS)
    fig_ber, axes_ber = plt.subplots(2, 3, figsize=(18, 10))
    axes_ber = axes_ber.flatten()

    fig_acc, axes_acc = plt.subplots(2, 3, figsize=(18, 10))
    axes_acc = axes_acc.flatten()

    summary = []

    for idx, cfg in enumerate(CONFIGS):
        label = cfg["label"]
        attack = cfg["attack"]
        lam = cfg["lam"]
        matches = find_matching_runs(runs, attack, lam)
        print(f"\n  [{idx+1}] {attack} λ={lam}: {len(matches)} runs found")

        # Collect honest and attacker BER per round across reps
        honest_ber_by_round = defaultdict(list)
        attacker_ber_by_round = defaultdict(list)
        all_exclusions = []
        all_accuracies = []

        for m in matches:
            rid = m["run_id"]
            train = load_train_data(rid)
            server = load_server_data(rid)
            if train is None:
                continue

            tp, fp, tn, fn, acc = compute_exclusion_accuracy(train)
            all_exclusions.append((tp, fp, tn, fn))
            all_accuracies.append(acc)
            print(f"    rep {rid}: TP={tp} FP={fp} TN={tn} FN={fn} acc={acc:.2%}")

            for rnd, clients in train.items():
                for pid, ber, is_att, excluded in clients:
                    if is_att:
                        attacker_ber_by_round[rnd].append(ber)
                    else:
                        honest_ber_by_round[rnd].append(ber)

        if not honest_ber_by_round:
            print(f"    WARNING: no data, skipping")
            continue

        # --- BER evolution plot ---
        ax = axes_ber[idx]
        common_rounds = sorted(set(honest_ber_by_round.keys()) & set(attacker_ber_by_round.keys()))

        if common_rounds:
            h_means = [np.mean(honest_ber_by_round[r]) for r in common_rounds]
            h_stds = [np.std(honest_ber_by_round[r]) for r in common_rounds]
            a_means = [np.mean(attacker_ber_by_round[r]) for r in common_rounds]
            a_stds = [np.std(attacker_ber_by_round[r]) for r in common_rounds]

            ax.plot(common_rounds, h_means, "C0-", linewidth=2, label="Honest")
            ax.fill_between(common_rounds,
                            [m - s for m, s in zip(h_means, h_stds)],
                            [m + s for m, s in zip(h_means, h_stds)],
                            color="C0", alpha=0.15)
            if any(a_means):  # only plot attacker if present
                ax.plot(common_rounds, a_means, "C3--", linewidth=2, label="Attacker")
                ax.fill_between(common_rounds,
                                [max(0, m - s) for m, s in zip(a_means, a_stds)],
                                [m + s for m, s in zip(a_means, a_stds)],
                                color="C3", alpha=0.15)

        ax.axhline(y=0.25, color="gray", linestyle=":", alpha=0.7, label="Threshold=0.25")
        ax.set_xlabel("FL Round")
        ax.set_ylabel("Watermark BER")
        ax.set_title(label, fontsize=11)
        ax.set_ylim(-0.05, 0.85)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

        # --- Accuracy plot ---
        ax_a = axes_acc[idx]
        accs_for_cfg = []
        for m in matches:
            server = load_server_data(m["run_id"])
            if server:
                accs_for_cfg.append(server)

        if accs_for_cfg:
            # Align and average
            min_len = min(len(a[1]) for a in accs_for_cfg)
            acc_matrix = np.array([a[1][:min_len] for a in accs_for_cfg])
            rounds = accs_for_cfg[0][0][:min_len]
            mean_acc = acc_matrix.mean(axis=0)
            std_acc = acc_matrix.std(axis=0)
            ax_a.plot(rounds, mean_acc, "C0-", linewidth=2, label="Accuracy")
            ax_a.fill_between(rounds, mean_acc - std_acc, mean_acc + std_acc,
                              color="C0", alpha=0.15)

        ax_a.set_xlabel("FL Round")
        ax_a.set_ylabel("Accuracy (%)")
        ax_a.set_title(label, fontsize=11)
        ax_a.set_ylim(0, 100)
        ax_a.grid(True, alpha=0.3)
        ax_a.legend(fontsize=8)

        # --- d' and AUC across all rounds ---
        all_honest_bers = []
        all_attacker_bers = []
        for rnd in common_rounds:
            all_honest_bers.extend(honest_ber_by_round[rnd])
            all_attacker_bers.extend(attacker_ber_by_round[rnd])

        dprime = compute_dprime(all_honest_bers, all_attacker_bers)
        auc = compute_auc(all_honest_bers, all_attacker_bers)
        mean_tp = np.mean([e[0] for e in all_exclusions]) if all_exclusions else 0
        mean_fp = np.mean([e[1] for e in all_exclusions]) if all_exclusions else 0
        mean_acc_def = np.mean(all_accuracies) if all_accuracies else 0

        summary.append({
            "label": label.replace("\n", " "),
            "attack": attack,
            "lambda": lam,
            "d_prime": dprime,
            "auc": auc,
            "defense_acc": mean_acc_def,
            "mean_tp": mean_tp,
            "mean_fp": mean_fp,
        })
        print(f"    d'={dprime:.4f} AUC={auc:.4f} defense_acc={mean_acc_def:.2%}")

    fig_ber.suptitle("BER Evolution — Honest vs Attacker Clients (per λ & attack)", fontsize=14, y=1.02)
    fig_ber.tight_layout()
    fig_ber.savefig(PLOTS_DIR / "eval_ber_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig_ber)
    print(f"\n  Saved: {PLOTS_DIR / 'eval_ber_evolution.png'}")

    fig_acc.suptitle("Server Accuracy — per λ & attack config", fontsize=14, y=1.02)
    fig_acc.tight_layout()
    fig_acc.savefig(PLOTS_DIR / "eval_accuracy.png", dpi=150, bbox_inches="tight")
    plt.close(fig_acc)
    print(f"  Saved: {PLOTS_DIR / 'eval_accuracy.png'}")

    # --- Summary bar chart ---
    fig_sum, axes_sum = plt.subplots(1, 3, figsize=(18, 5))
    labels = [s["label"] for s in summary]
    x = np.arange(len(labels))

    # d'-score
    dprimes = [s["d_prime"] for s in summary]
    colors = ["C0" if np.isfinite(d) and d > 0 else "C3" for d in dprimes]
    axes_sum[0].bar(x, [d if np.isfinite(d) else 0 for d in dprimes], color=colors)
    axes_sum[0].set_ylabel("d'-score")
    axes_sum[0].set_title("Watermark Sensitivity (d')")
    axes_sum[0].set_xticks(x)
    axes_sum[0].set_xticklabels(labels, fontsize=8)
    axes_sum[0].axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    axes_sum[0].grid(True, alpha=0.3, axis="y")

    # ROC-AUC
    aucs = [s["auc"] for s in summary]
    colors = ["C0" if np.isfinite(a) and a > 0.5 else "C3" for a in aucs]
    axes_sum[1].bar(x, [a if np.isfinite(a) else 0.5 for a in aucs], color=colors)
    axes_sum[1].set_ylabel("ROC-AUC")
    axes_sum[1].set_title("Attack Detection (ROC-AUC)")
    axes_sum[1].set_xticks(x)
    axes_sum[1].set_xticklabels(labels, fontsize=8)
    axes_sum[1].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="random")
    axes_sum[1].legend()
    axes_sum[1].grid(True, alpha=0.3, axis="y")

    # Defense accuracy
    def_accs = [s["defense_acc"] for s in summary]
    axes_sum[2].bar(x, def_accs, color="C2")
    axes_sum[2].set_ylabel("Defense Accuracy")
    axes_sum[2].set_title("Exclusion Accuracy (TP+TN)/(all)")
    axes_sum[2].set_xticks(x)
    axes_sum[2].set_xticklabels(labels, fontsize=8)
    axes_sum[2].set_ylim(0, 1)
    axes_sum[2].grid(True, alpha=0.3, axis="y")

    fig_sum.suptitle("Semi-Fragility Evaluation Summary", fontsize=14, y=1.02)
    fig_sum.tight_layout()
    fig_sum.savefig(PLOTS_DIR / "eval_defense_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig_sum)
    print(f"  Saved: {PLOTS_DIR / 'eval_defense_summary.png'}")

    # Print summary table
    print("\n  === SUMMARY TABLE ===")
    print(f"  {'Config':<30} {'d-prime':>8} {'AUC':>8} {'Def.Acc':>10}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*10}")
    for s in summary:
        dp = f"{s['d_prime']:.4f}" if np.isfinite(s['d_prime']) else "N/A"
        au = f"{s['auc']:.4f}" if np.isfinite(s['auc']) else "N/A"
        da = f"{s['defense_acc']:.2%}"
        print(f"  {s['label']:<30} {dp:>8} {au:>8} {da:>10}")


if __name__ == "__main__":
    main()
