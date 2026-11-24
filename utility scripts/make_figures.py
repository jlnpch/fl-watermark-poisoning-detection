#!/usr/bin/env python3
"""Generate publication figures (PDFs) for the paper.

Figure 2 — BER separation (sign-flip vs freeze-FC).
Figure 3 — Complementarity (freeze-FC final accuracy across defenses).
Figure 4 — Coverage curve (K-scaling).

Locates run IDs from results/run_registry.jsonl and aggregates across 3 seeds.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "/home2/julian/test/fl-watermark-poisoning-detection/results"
ROOT = "/home2/julian/test/fl-watermark-poisoning-detection"

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
})


def load_registry():
    entries = []
    for line in open(os.path.join(RESULTS, "run_registry.jsonl")):
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception:
            pass
    return entries


ENTRIES = load_registry()


def _scale(e):
    try:
        return float(e.get("sign-flip-scale", 0))
    except (TypeError, ValueError):
        return 0.0


def _seed(e):
    try:
        return int(e.get("seed", -1))
    except (TypeError, ValueError):
        return -1


def _single_fc(e):
    wl = e.get("watermark-layers") or ""
    return wl in ("", "backbone.fc.weight")


def find_runs(attacker, defense, scale, seeds, frozen=None, single_fc=True):
    """Return {seed: run_id} matching the filters."""
    out = {}
    for e in ENTRIES:
        if e.get("attacker-type") != attacker:
            continue
        if e.get("defense-method") != defense:
            continue
        if abs(_scale(e) - scale) > 1e-9:
            continue
        s = _seed(e)
        if s not in seeds:
            continue
        if single_fc and not _single_fc(e):
            continue
        fl = e.get("frozen-layers") or ""
        if frozen is not None and fl != frozen:
            continue
        out[s] = e["run_id"]
    return out


def read_train(rid):
    """Return {round: {'honest': [ber...], 'attacker': [ber...]}}."""
    data = {}
    with open(os.path.join(RESULTS, f"run_{rid}_train.csv")) as f:
        for r in csv.DictReader(f):
            rnd = int(r["round"])
            ber = float(r["watermark_ber"])
            key = "attacker" if int(r["is_attacker"]) == 1 else "honest"
            data.setdefault(rnd, {"honest": [], "attacker": []})[key].append(ber)
    return data


def read_server_acc(rid):
    with open(os.path.join(RESULTS, f"run_{rid}_server.csv")) as f:
        rows = list(csv.DictReader(f))
    return float(rows[-1]["server_acc"])


def _mad_threshold(bes):
    bes = np.array(bes)
    med = np.median(bes)
    mad = np.median(np.abs(bes - med))
    return med + 4.0 * mad


# --------------------------------------------------------------------------- #
# Figure 2 — BER separation
# --------------------------------------------------------------------------- #
def figure2():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)

    configs = [
        ("(a) naive sign-flip (s2)", "sign_flip", "mad", "blue", "red"),
        ("(b) freeze-and-poison (s2)", "freeze_fc", "mad", "blue", "red"),
    ]
    used = {}
    for ax, (title, attacker, defense, hon_c, att_c) in zip(axes, configs):
        seeds = find_runs(attacker, defense, 2.0, {0, 1, 2}, single_fc=True)
        used[attacker] = seeds
        rounds = range(1, 51)
        hon_mean, hon_std, att_mean = [], [], []
        for rnd in rounds:
            hon_all, att_all = [], []
            for s in (0, 1, 2):
                rid = seeds.get(s)
                if rid is None:
                    continue
                d = read_train(rid)
                if rnd in d:
                    hon_all.extend(d[rnd]["honest"])
                    att_all.extend(d[rnd]["attacker"])
            hon_mean.append(np.mean(hon_all) if hon_all else np.nan)
            hon_std.append(np.std(hon_all) if hon_all else 0.0)
            att_mean.append(np.mean(att_all) if att_all else np.nan)

        r = np.array(list(rounds))
        hm = np.array(hon_mean); hs = np.array(hon_std); am = np.array(att_mean)
        ax.plot(r, am, color=att_c, lw=1.6, label="attacker")
        ax.plot(r, hm, color=hon_c, lw=1.6, label="honest (mean)")
        ax.fill_between(r, hm - hs, hm + hs, color=hon_c, alpha=0.25, label="honest ±std")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("FL round")
        ax.set_ylim(0, 0.5)
        ax.legend(loc="upper right", frameon=False)
        ax.grid(alpha=0.25, lw=0.4)

    axes[0].set_ylabel("BER")
    fig.tight_layout()
    out = os.path.join(ROOT, "fig_ber_separation.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, used


# --------------------------------------------------------------------------- #
# Figure 3 — Complementarity
# --------------------------------------------------------------------------- #
def figure3():
    defenses = ["none", "mad", "krum", "fltrust", "trimmed-mean", "median"]
    scales = [2.0, 5.0]
    means, stds = {}, {}
    for d in defenses:
        for sc in scales:
            seeds = find_runs("freeze_fc", d, sc, {0, 1, 2}, frozen="backbone.fc.weight", single_fc=True)
            accs = [read_server_acc(rid) for rid in seeds.values() if os.path.exists(os.path.join(RESULTS, f"run_{rid}_server.csv"))]
            means[(d, sc)] = np.mean(accs) if accs else np.nan
            stds[(d, sc)] = np.std(accs) if accs else 0.0

    x = np.arange(len(defenses))
    w = 0.36
    fig, ax = plt.subplots(figsize=(6.0, 2.8))
    b1 = ax.bar(x - w / 2, [means[(d, 2.0)] for d in defenses], w, yerr=[stds[(d, 2.0)] for d in defenses],
                color="C0", capsize=2, label="sign-flip s2")
    b2 = ax.bar(x + w / 2, [means[(d, 5.0)] for d in defenses], w, yerr=[stds[(d, 5.0)] for d in defenses],
                color="C3", capsize=2, label="sign-flip s5")
    ax.set_xticks(x)
    ax.set_xticklabels(defenses, rotation=30, ha="right")
    ax.set_ylabel("final accuracy")
    ax.set_ylim(0, 1.0)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25, lw=0.4)
    fig.tight_layout()
    out = os.path.join(ROOT, "fig_complementarity.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- #
# Figure 4 — Coverage curve
# --------------------------------------------------------------------------- #
def figure4():
    Ks = list(range(1, 6))
    acc_mean, acc_std = [], []
    ber_mean, ber_std = [], []
    for K in Ks:
        if K == 1:
            seeds = find_runs("freeze_fc", "mad", 2.0, {0, 1, 2}, frozen="backbone.fc.weight", single_fc=True)
        else:
            seeds = {}
            for e in ENTRIES:
                if e.get("attacker-type") != "freeze_fc" or e.get("defense-method") != "mad":
                    continue
                if abs(_scale(e) - 2.0) > 1e-9:
                    continue
                if _seed(e) not in {0, 1, 2}:
                    continue
                fl = e.get("frozen-layers") or ""
                if fl.count(",") + 1 == K:
                    seeds[_seed(e)] = e["run_id"]
        rids = [rid for rid in seeds.values() if os.path.exists(os.path.join(RESULTS, f"run_{rid}_server.csv"))]
        accs = [read_server_acc(rid) for rid in rids]
        acc_mean.append(np.mean(accs)); acc_std.append(np.std(accs))

        bes = []
        for rid in rids:
            d = read_train(rid)
            for rnd in d.values():
                bes.extend(rnd["honest"])
        ber_mean.append(np.mean(bes)); ber_std.append(np.std(bes))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax1.errorbar(Ks, acc_mean, yerr=acc_std, marker="o", capsize=2, color="C0", lw=1.4)
    ax1.set_xlabel("K (watermarked layers)")
    ax1.set_ylabel("final accuracy")
    ax1.set_ylim(0.3, 0.75)
    ax1.set_xticks(Ks)
    ax1.grid(alpha=0.25, lw=0.4)

    ax2.errorbar(Ks, ber_mean, yerr=ber_std, marker="s", capsize=2, color="C3", lw=1.4)
    ax2.set_xlabel("K (watermarked layers)")
    ax2.set_ylabel("honest max-BER")
    ax2.set_ylim(0, 0.5)
    ax2.set_xticks(Ks)
    ax2.grid(alpha=0.25, lw=0.4)

    fig.tight_layout()
    out = os.path.join(ROOT, "fig_coverage.pdf")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    p2, used2 = figure2()
    p3 = figure3()
    p4 = figure4()
    print("Generated:")
    print("  ", p2)
    print("  ", p3)
    print("  ", p4)
    print("\nFigure 2 run IDs:")
    for attacker, seeds in used2.items():
        print(f"  {attacker}: {dict(sorted(seeds.items()))}")
