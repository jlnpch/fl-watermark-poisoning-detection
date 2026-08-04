#!/usr/bin/env python3
"""Plot experiments: 128-bit watermark and pretrain LR-scheduler sweeps.

Produces one 5-panel figure per experiment, saved under
results/plots/various_experiments/{experiment}/evolution.png:
  1. watermark regularization loss vs pretrain epoch (log)
  2. task loss vs pretrain epoch (log)
  3. watermark BER vs pretrain epoch
  4. watermark BER vs FL round (mean over clients)
  5. server accuracy vs FL round (%)

Experiments:
  --experiment 128bit   -> watermark-num-bits=128 (standard scheduler off)
  --experiment lr_sched -> pretrain-lr-scheduler=true (standard 64-bit)

Filter: clean baseline (no attacker), 5000 server samples, N pretrain epochs.
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

EXPERIMENTS = {
    "128bit": {"dir": "128bit_watermark", "title": "128-bit watermark (baseline sweep)"},
    "lr_sched": {"dir": "lr_scheduler", "title": "64-bit watermark + cosine LR scheduler (baseline sweep)"},
}


def parse_args():
    parser = argparse.ArgumentParser(description="Experiment plot script")
    parser.add_argument("--experiment", required=True, choices=sorted(EXPERIMENTS),
                        help="Which experiment to plot")
    parser.add_argument("--pretrain-epochs", type=int, default=50,
                        help="Pretrain epochs of the config (default: 50)")
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


def experiment_filter(experiment):
    """Return predicate for runs belonging to this experiment."""
    if experiment == "128bit":
        return lambda r: (r.get("watermark-num-bits", 64) == 128
                          and not r.get("pretrain-lr-scheduler", False))
    return lambda r: (r.get("watermark-num-bits", 64) == 64
                      and r.get("pretrain-lr-scheduler", False))


def select_runs(runs, experiment, ep):
    pred = experiment_filter(experiment)
    matches = [r for r in runs
               if r.get("attacker-type", "none") == "none"
               and r.get("attacker-fraction", 0) == 0
               and r.get("server-private-samples") == 5000
               and r.get("pretrain-epochs") == ep
               and r.get("client-samples") == 4000
               and not r.get("pretrain-only", False)
               and pred(r)]
    seen = set()
    out = []
    for r in matches:
        if r["run_id"] in seen:
            continue
        seen.add(r["run_id"])
        out.append(r)
    return out


def load_pretrain(run_id):
    path = RESULTS_DIR / f"run_{run_id}_pretrain.csv"
    if not path.exists():
        return None
    rows = list(csv.DictReader(path.open()))
    return {
        "epoch": [int(r["epoch"]) for r in rows],
        "task_loss": [float(r["task_loss"]) for r in rows],
        "watermark_loss": [float(r["watermark_loss"]) for r in rows],
        "watermark_ber": [float(r["watermark_ber"]) for r in rows],
    }


def load_train_ber(run_id):
    """Per-round mean watermark BER over all clients."""
    path = RESULTS_DIR / f"run_{run_id}_train.csv"
    if not path.exists():
        return None
    by_round = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            by_round[int(row["round"])].append(float(row["watermark_ber"]))
    rounds = sorted(by_round)
    return rounds, [np.mean(by_round[r]) for r in rounds]


def load_server_acc(run_id):
    path = RESULTS_DIR / f"run_{run_id}_server.csv"
    if not path.exists():
        return None
    rounds, accs = [], []
    with path.open() as f:
        for row in csv.DictReader(f):
            rounds.append(int(row["round"]))
            accs.append(float(row["server_acc"]) * 100)
    return rounds, accs


def main():
    args = parse_args()
    ep = args.pretrain_epochs
    exp_info = EXPERIMENTS[args.experiment]

    runs = load_registry()
    matched = select_runs(runs, args.experiment, ep)
    print(f"Found {len(matched)} experiment runs for {args.experiment} ({ep}ep)")
    if not matched:
        print("No runs found — have the experiment sweeps finished?")
        sys.exit(1)

    # Group by lambda
    lam_runs = defaultdict(list)
    for r in matched:
        lam = round(r.get("watermark-lambda", 0), 3)
        lam_runs[lam].append(r)
    active = [lam for lam in LAMBDAS if lam_runs[lam]]

    colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(active)))

    fig, axes = plt.subplots(1, 5, figsize=(24, 5.5))
    titles = ["Watermark loss (pretrain)", "Task loss (pretrain)", "BER (pretrain)",
              "BER (FL rounds)", "Accuracy (FL rounds)"]

    for lam, color in zip(active, colors):
        reps = lam_runs[lam]

        # --- Pretrain panels ---
        pretrains = [load_pretrain(r["run_id"]) for r in reps]
        pretrains = [p for p in pretrains if p]
        if pretrains:
            min_e = max(p["epoch"][0] for p in pretrains)
            max_e = min(p["epoch"][-1] for p in pretrains)
            common = list(range(min_e, max_e + 1))
            for p in pretrains:
                p = {k: np.array(v) for k, v in p.items()}
                imap = dict(zip(p["epoch"], range(len(p["epoch"]))))
                common_idx = [imap[e] for e in common]
                p["_idx"] = common_idx
            wm = np.array([[p["watermark_loss"][i] for i in p["_idx"]] for p in pretrains])
            tk = np.array([[p["task_loss"][i] for i in p["_idx"]] for p in pretrains])
            br = np.array([[p["watermark_ber"][i] for i in p["_idx"]] for p in pretrains])

            for ax, mat in ((axes[0], wm), (axes[1], tk)):
                logmat = np.log(np.clip(mat, 1e-12, None))
                ax.plot(common, np.exp(np.mean(logmat, axis=0)), "-", color=color,
                        linewidth=2, label=f"λ={lam}")
                ax.fill_between(common,
                                np.exp(np.mean(logmat, axis=0) - np.std(logmat, axis=0)),
                                np.exp(np.mean(logmat, axis=0) + np.std(logmat, axis=0)),
                                color=color, alpha=0.15)
            axes[2].plot(common, np.mean(br, axis=0), "-", color=color, linewidth=2, label=f"λ={lam}")
            axes[2].fill_between(common, np.mean(br, axis=0) - np.std(br, axis=0),
                                 np.mean(br, axis=0) + np.std(br, axis=0), color=color, alpha=0.15)

        # --- FL panels ---
        fl_bers = [load_train_ber(r["run_id"]) for r in reps]
        fl_bers = [b for b in fl_bers if b]
        if fl_bers:
            rounds = fl_bers[0][0]
            mat = np.array([b[1][:len(rounds)] for b in fl_bers])
            axes[3].plot(rounds, np.mean(mat, axis=0), "-", color=color, linewidth=2)
            axes[3].fill_between(rounds, np.mean(mat, axis=0) - np.std(mat, axis=0),
                                 np.mean(mat, axis=0) + np.std(mat, axis=0), color=color, alpha=0.15)

        accs = [load_server_acc(r["run_id"]) for r in reps]
        accs = [a for a in accs if a]
        if accs:
            rounds = accs[0][0]
            mat = np.array([a[1][:len(rounds)] for a in accs])
            axes[4].plot(rounds, np.mean(mat, axis=0), "-", color=color, linewidth=2)
            axes[4].fill_between(rounds, np.mean(mat, axis=0) - np.std(mat, axis=0),
                                 np.mean(mat, axis=0) + np.std(mat, axis=0), color=color, alpha=0.15)

    for ax, title in zip(axes, titles):
        ax.set_xlabel("Epoch / Round")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[0].set_yscale("log")
    axes[1].set_yscale("log")
    axes[2].set_ylabel("BER")
    axes[3].set_ylabel("BER (mean over clients)")
    axes[4].set_ylabel("Accuracy (%)")
    axes[4].set_ylim(0, 100)
    axes[0].legend(fontsize=8, ncol=2)
    axes[2].legend(fontsize=8)

    out_dir = RESULTS_DIR / "plots" / "various_experiments" / exp_info["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.suptitle(f"{exp_info['title']} — 5000 samples, {ep} pretrain epochs", fontsize=14, y=1.03)
    fig.tight_layout()
    path = out_dir / "evolution.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    # Summary table (final pretrain values + final FL BER/acc)
    print("\n  === SUMMARY (mean over reps) ===")
    print(f"  {'λ':<8} {'N':>3} {'wm(pret)':>9} {'task(pret)':>10} {'BER(pret)':>9} {'BER(FL end)':>11} {'acc(FL end)':>10}")
    print(f"  {'-'*8} {'-'*3} {'-'*9} {'-'*10} {'-'*9} {'-'*11} {'-'*10}")
    gmean = lambda v: float(np.exp(np.mean(np.log(np.clip(v, 1e-12, None)))))
    for lam in active:
        reps = lam_runs[lam]
        n = len(reps)
        prets = [p for p in (load_pretrain(r["run_id"]) for r in reps) if p]
        wml = gmean([p["watermark_loss"][-1] for p in prets]) if prets else float("nan")
        tl = gmean([p["task_loss"][-1] for p in prets]) if prets else float("nan")
        bpret = np.mean([p["watermark_ber"][-1] for p in prets]) if prets else float("nan")
        fbers = [b for b in (load_train_ber(r["run_id"]) for r in reps) if b]
        ber_end = np.mean([b[1][-1] for b in fbers]) if fbers else float("nan")
        accs = [a for a in (load_server_acc(r["run_id"]) for r in reps) if a]
        acc_end = np.mean([a[1][-1] for a in accs]) if accs else float("nan")
        print(f"  {lam:<8} {n:>3} {wml:>9.5f} {tl:>10.5f} {bpret:>9.4f} {ber_end:>11.4f} {acc_end:>10.2f}")


if __name__ == "__main__":
    main()
