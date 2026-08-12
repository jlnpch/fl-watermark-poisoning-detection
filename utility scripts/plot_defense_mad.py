#!/usr/bin/env python3
"""Plot MAD-defense results vs sign-flip attack (scale 5 and 2), alpha sweep.

Generates (from results/run_registry.jsonl, MAD runs, 5000/50ep, λ=0.01, 128-bit):
1. mad_ber_evolution.png — BER evolution per (scale, alpha): honest vs attacker
   mean±std over reps, plus the recomputed dynamic threshold line (median+alpha*MAD)
2. mad_defense_summary.png — detection rate (TPR), false-positive rate (FPR),
   d'-score, AUC, and final FL accuracy vs alpha for each scale
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

ALPHAS = [2.0, 3.0, 4.0, 7.0]
SCALES = [5, 2]
PRETRAIN_EPOCHS = 50
LAMBDA = 0.01


def parse_args():
    p = argparse.ArgumentParser(description="MAD-defense alpha-sweep plots")
    p.add_argument("--pretrain-epochs", type=int, default=PRETRAIN_EPOCHS)
    return p.parse_args()


def load_registry():
    reg_path = RESULTS_DIR / "run_registry.jsonl"
    if not reg_path.exists():
        print(f"ERROR: {reg_path} not found")
        sys.exit(1)
    runs = []
    for line in open(reg_path):
        line = line.strip()
        if line:
            runs.append(json.loads(line))
    return runs


def find_mad_runs(runs, scale, alpha, pretrain_epochs):
    matched = [r for r in runs
               if r.get("defense-method") == "mad"
               and abs(r.get("mad-alpha", 0) - alpha) < 1e-9
               and r.get("attacker-type") == "sign_flip"
               and abs(r.get("sign-flip-scale", 0) - scale) < 1e-9
               and abs(r.get("watermark-lambda", 0) - LAMBDA) < 1e-9
               and r.get("attacker-fraction", 0) == 0.1
               and r.get("server-private-samples") == 5000
               and r.get("pretrain-epochs") == pretrain_epochs
               and not r.get("pretrain-only", False)
               and Path(RESULTS_DIR / f"run_{r['run_id']}_train.csv").exists()]
    matched.sort(key=lambda r: r.get("timestamp", ""))
    return matched


def load_train_data(run_id):
    path = RESULTS_DIR / f"run_{run_id}_train.csv"
    if not path.exists():
        return None
    data = defaultdict(list)
    with open(path) as f:
        for row in csv.DictReader(f):
            rnd = int(row["round"])
            data[rnd].append((int(row["partition_id"]),
                              float(row["watermark_ber"]),
                              int(row["is_attacker"]),
                              int(row["excluded"])))
    return data


def load_server_data(run_id):
    path = RESULTS_DIR / f"run_{run_id}_server.csv"
    if not path.exists():
        return None
    rounds, accs = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            accs.append(float(row["server_acc"]) * 100)
    return rounds, accs


def mad_threshold(bes, alpha):
    bes = np.asarray(bes, dtype=float)
    if len(bes) < 2:
        return np.inf
    med = np.median(bes)
    mad = np.median(np.abs(bes - med))
    return med + alpha * mad


def compute_dprime(honest_bers, attacker_bers):
    honest = np.array(honest_bers)
    attacker = np.array(attacker_bers)
    if len(honest) < 2 or len(attacker) < 2:
        return np.nan
    std_pool = np.sqrt((np.var(honest, ddof=1) + np.var(attacker, ddof=1)) / 2)
    if std_pool < 1e-12:
        return np.nan
    return (np.mean(attacker) - np.mean(honest)) / std_pool


def compute_auc(honest_bers, attacker_bers):
    honest = np.array(honest_bers)
    attacker = np.array(attacker_bers)
    if len(honest) < 2 or len(attacker) < 2:
        return np.nan
    scores = np.concatenate([honest, attacker])
    labels = np.concatenate([np.zeros(len(honest)), np.ones(len(attacker))])
    order = np.argsort(-scores)
    scores, labels = scores[order], labels[order]
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)
    tpr = tps / n_pos
    fpr = fps / n_neg
    return np.trapz(tpr, fpr)


def main(pretrain_epochs=PRETRAIN_EPOCHS):
    runs = load_registry()
    print(f"Found {len(runs)} runs in registry")
    plots_dir = RESULTS_DIR / "plots" / f"mad_sweep_{pretrain_epochs}ep"
    plots_dir.mkdir(parents=True, exist_ok=True)

    n_rows = len(SCALES)
    n_cols = len(ALPHAS)
    fig_ber, axes_ber = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes_ber = np.atleast_2d(axes_ber)

    summary_rows = []

    for si, scale in enumerate(SCALES):
        for ai, alpha in enumerate(ALPHAS):
            matches = find_mad_runs(runs, scale, alpha, pretrain_epochs)
            print(f"\n  scale={scale} alpha={alpha}: {len(matches)} runs")

            honest_by_round = defaultdict(list)
            attacker_by_round = defaultdict(list)
            excluded_by_round = defaultdict(list)  # rnd -> [1 if excluded else 0 for each client]
            is_att_by_round = defaultdict(list)    # rnd -> [1 if attacker else 0 for each client]
            thr_by_round = defaultdict(list)
            all_accs = []

            for m in matches:
                rid = m["run_id"]
                train = load_train_data(rid)
                server = load_server_data(rid)
                if train is None:
                    continue

                for rnd, clients in train.items():
                    bes = [ber for _, ber, _, _ in clients]
                    thr_by_round[rnd].append(mad_threshold(bes, alpha))
                    for pid, ber, is_att, excluded in clients:
                        (attacker_by_round if is_att else honest_by_round)[rnd].append(ber)
                        excluded_by_round[rnd].append(excluded)
                        is_att_by_round[rnd].append(is_att)

                if server:
                    all_accs.append(server[1])

            if not honest_by_round:
                print(f"    WARNING: no data, skipping")
                continue

            # Per-round exclusion stats (mean over reps)
            rounds = sorted(honest_by_round.keys())
            tpr_by_rnd, fpr_by_rnd = [], []
            for rnd in rounds:
                ex = np.array(excluded_by_round[rnd])
                att = np.array(is_att_by_round[rnd])
                n_att = att.sum()
                n_hon = len(att) - n_att
                if n_att > 0:
                    tpr_by_rnd.append((ex[att == 1].sum() / n_att) * 100)
                if n_hon > 0:
                    fpr_by_rnd.append((ex[att == 0].sum() / n_hon) * 100)

            tpr = np.mean(tpr_by_rnd) if tpr_by_rnd else np.nan
            fpr = np.mean(fpr_by_rnd) if fpr_by_rnd else np.nan

            # Aggregate BERs for d' and AUC (all rounds, all reps)
            all_honest = [b for rnd in rounds for b in honest_by_round[rnd]]
            all_attacker = [b for rnd in rounds for b in attacker_by_round[rnd]]
            dprime = compute_dprime(all_honest, all_attacker)
            auc = compute_auc(all_honest, all_attacker)

            final_acc = np.mean([a[-1] for a in all_accs]) if all_accs else np.nan
            summary_rows.append((scale, alpha, tpr, fpr, dprime, auc, final_acc, len(matches)))

            # --- BER evolution subplot ---
            ax = axes_ber[si, ai]
            h_means = [np.mean(honest_by_round[r]) for r in rounds]
            h_stds = [np.std(honest_by_round[r]) for r in rounds]
            ax.plot(rounds, h_means, "C0-", linewidth=2, label="Honest")
            ax.fill_between(rounds, [m - s for m, s in zip(h_means, h_stds)],
                            [m + s for m, s in zip(h_means, h_stds)],
                            color="C0", alpha=0.15)

            common = sorted(set(rounds) & set(attacker_by_round.keys()))
            if common:
                a_means = [np.mean(attacker_by_round[r]) for r in common]
                a_stds = [np.std(attacker_by_round[r]) for r in common]
                ax.plot(common, a_means, "C3--", linewidth=2, label="Attacker")
                ax.fill_between(common,
                                [max(0, m - s) for m, s in zip(a_means, a_stds)],
                                [m + s for m, s in zip(a_means, a_stds)],
                                color="C3", alpha=0.15)

            if thr_by_round:
                t_rounds = sorted(thr_by_round.keys())
                t_means = [np.mean(thr_by_round[r]) for r in t_rounds]
                ax.plot(t_rounds, t_means, "C1-", linewidth=1.5,
                        label=f"Threshold (median+{alpha}·MAD)")

            ax.set_xlabel("FL Round")
            ax.set_ylabel("Watermark BER")
            ax.set_title(f"sign-flip scale={scale}, α={alpha}  (TPR {tpr:.0f}%, FPR {fpr:.0f}%)",
                         fontsize=11)
            ax.set_ylim(-0.05, 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig_ber.suptitle(f"MAD-defense BER evolution (5000/{pretrain_epochs}ep, λ={LAMBDA}, 128-bit, 3 reps)",
                     fontsize=13)
    fig_ber.tight_layout(rect=[0, 0, 1, 0.97])
    ber_path = plots_dir / "mad_ber_evolution.png"
    fig_ber.savefig(ber_path, dpi=150)
    plt.close(fig_ber)
    print(f"\nSaved {ber_path}")

    # --- Summary figure: metrics vs alpha for each scale ---
    n_metrics = 2  # exclusion rates + final accuracy
    fig_sum, axes_sum = plt.subplots(n_rows, n_metrics, figsize=(6 * n_metrics, 5 * n_rows))
    axes_sum = np.atleast_2d(axes_sum)

    for si, scale in enumerate(SCALES):
        rows = [r for r in summary_rows if r[0] == scale]
        alphas = [r[1] for r in rows]
        tprs = [r[2] for r in rows]
        fprs = [r[3] for r in rows]
        final_accs = [r[6] for r in rows]

        ax = axes_sum[si, 0]
        ax.plot(alphas, tprs, "C2-o", linewidth=2, label="TPR (detection)")
        ax.plot(alphas, fprs, "C3-o", linewidth=2, label="FPR (honest excluded)")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Rate (%)")
        ax.set_title(f"scale={scale}: exclusion rates vs alpha")
        ax.set_ylim(-5, 105)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

        ax = axes_sum[si, 1]
        ax.plot(alphas, final_accs, "C1-o", linewidth=2, label="Final accuracy")
        ax.set_xlabel("alpha")
        ax.set_ylabel("Accuracy (%)")
        ax.set_ylim(0, 100)
        ax.set_title(f"scale={scale}: final FL accuracy vs alpha")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    fig_sum.suptitle("MAD-defense summary: effect of alpha", fontsize=13)
    fig_sum.tight_layout(rect=[0, 0, 1, 0.97])
    sum_path = plots_dir / "mad_defense_summary.png"
    fig_sum.savefig(sum_path, dpi=150)
    plt.close(fig_sum)
    print(f"Saved {sum_path}")

    # --- Accuracy evolution per run, one subplot per (scale, alpha) ---
    fig_acc, axes_acc = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axes_acc = np.atleast_2d(axes_acc)

    for si, scale in enumerate(SCALES):
        for ai, alpha in enumerate(ALPHAS):
            matches = find_mad_runs(runs, scale, alpha, pretrain_epochs)
            ax = axes_acc[si, ai]
            rep_accs = []
            for m in matches:
                server = load_server_data(m["run_id"])
                if server:
                    rounds, accs = server
                    rep_accs.append(accs)
                    ax.plot(rounds, accs, linewidth=1.2, alpha=0.6)
            if rep_accs:
                min_len = min(len(a) for a in rep_accs)
                mean_acc = np.mean([a[:min_len] for a in rep_accs], axis=0)
                std_acc = np.std([a[:min_len] for a in rep_accs], axis=0)
                r = rounds[:min_len]
                ax.plot(r, mean_acc, "k-", linewidth=2.5, label="mean")
                ax.fill_between(r, mean_acc - std_acc, mean_acc + std_acc,
                                color="k", alpha=0.1)
            ax.axhline(74, color="gray", linestyle=":", alpha=0.7, label="no-attack baseline")
            ax.set_xlabel("FL Round")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"sign-flip scale={scale}, α={alpha} (n={len(matches)})", fontsize=11)
            ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

    fig_acc.suptitle("FL accuracy evolution — MAD defense, all runs (3 reps each)",
                     fontsize=13)
    fig_acc.tight_layout(rect=[0, 0, 1, 0.97])
    acc_path = plots_dir / "mad_acc_evolution.png"
    fig_acc.savefig(acc_path, dpi=150)
    plt.close(fig_acc)
    print(f"Saved {acc_path}")

    # --- Baseline (no attacker) FP analysis ---
    base_runs = [r for r in runs
                 if r.get("defense-method") == "mad"
                 and r.get("attacker-type") == "none"
                 and abs(r.get("watermark-lambda", 0) - LAMBDA) < 1e-9
                 and r.get("server-private-samples") == 5000
                 and r.get("pretrain-epochs") == pretrain_epochs
                 and not r.get("pretrain-only", False)
                 and Path(RESULTS_DIR / f"run_{r['run_id']}_train.csv").exists()]

    base_stats = defaultdict(list)  # alpha -> [(fp_rounds, fp_total, n_hon)]
    for r in base_runs:
        train = load_train_data(r["run_id"])
        if train is None:
            continue
        fp_rounds = sum(1 for rnd, c in train.items() if any(e and not a for _, _, a, e in c))
        fp_total = sum(1 for rnd, c in train.items() for _, _, a, e in c if e and not a)
        n_hon = sum(len([x for x in c if not x[2]]) for c in train.values())
        base_stats[r["mad-alpha"]].append((fp_rounds, fp_total, n_hon))

    if base_stats:
        alphas = sorted(base_stats)
        fpr_means = [np.mean([x[1] for x in base_stats[a]]) / np.mean([x[2] for x in base_stats[a]]) * 100
                     for a in alphas]
        fpr_stds = [np.std([x[1] / x[2] * 100 for x in base_stats[a]]) for a in alphas]

        fig_base, ax_base = plt.subplots(1, 1, figsize=(7, 5))
        x = np.arange(len(alphas))
        ax_base.bar(x, fpr_means, 0.5, yerr=fpr_stds, color="C0", label="FPR/client-round")
        ax_base.set_xticks(x, [str(a) for a in alphas])
        ax_base.set_xlabel("alpha")
        ax_base.set_ylabel("FPR (%)")
        ax_base.set_title("Baseline (no attacker): client false-positive rate")
        ax_base.grid(True, axis="y", alpha=0.3)
        ax_base.legend(fontsize=9)
        fig_base.suptitle(f"Baseline FP (5000/{pretrain_epochs}ep, no attacker, 3 reps)",
                          fontsize=13)
        fig_base.tight_layout(rect=[0, 0, 1, 0.95])
        base_path = plots_dir / "mad_baseline_fp.png"
        fig_base.savefig(base_path, dpi=150)
        plt.close(fig_base)
        print(f"Saved {base_path}")

        print("\nBaseline (no attacker) FP summary:")
        print("{:<6}{:>16}{:>16}{:>12}".format("alpha", "FP rounds/50", "FPR/client%", "n"))
        for a in alphas:
            rows = base_stats[a]
            print("{:<6}{:>12.1f} ± {:<4.1f}{:>13.1f} ± {:<4.1f}{:>12}".format(
                a,
                np.mean([x[0] for x in rows]), np.std([x[0] for x in rows]),
                np.mean([x[1] / x[2] * 100 for x in rows]),
                np.std([x[1] / x[2] * 100 for x in rows]),
                len(rows)))

    # --- Text summary table ---
    print("\n{:<6}{:<6}{:>8}{:>8}{:>8}{:>8}{:>10}{:>6}".format(
        "scale", "alpha", "TPR%", "FPR%", "d'", "AUC", "finalAcc%", "n"))
    for scale, alpha, tpr, fpr, dprime, auc, final_acc, n in summary_rows:
        print("{:<6}{:<6}{:>8.1f}{:>8.2f}{:>8.2f}{:>8.3f}{:>10.1f}{:>6}".format(
            scale, alpha, tpr, fpr, dprime, auc, final_acc, n))


if __name__ == "__main__":
    main(pretrain_epochs=parse_args().pretrain_epochs)
