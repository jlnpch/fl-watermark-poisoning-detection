#!/usr/bin/env python3
"""Pillar A sweep analysis.

Reads results/logs/pillar_a_summary.log (tag -> defense/attack/scale/seed/run_id),
then computes:
  1. final accuracy (round 50, and last-5-round mean) per (defense x attack) cell
  2. TPR/FPR for mad and krum (from train.csv excluded vs is_attacker)
  3. federated wall-clock per defense (from per-run logs)
  4. run-id mapping / dedup check against run_registry.jsonl
"""
import collections
import csv
import json
import os
import re
import statistics

RESULTS = "/home2/julian/test/fl-watermark-poisoning-detection/results"
SUMMARY = os.path.join(RESULTS, "logs", "pillar_a_summary.log")

DEFENSES = ["none", "mad", "fltrust", "krum", "trimmed-mean", "median"]
# cell -> (summary attack, summary scale)
CELLS = {
    "clean": ("clean", "none"),
    "s2": ("sign_flip", "2.0"),
    "s5": ("sign_flip", "5.0"),
    "noise": ("noise", "1.0"),
}


def load_runs():
    """Return {(defense, attack, scale, seed): run_id} — last entry wins on dup."""
    runs = {}
    with open(SUMMARY) as f:
        for line in f:
            if "\t" not in line.strip():
                continue
            parts = line.strip().split("\t")
            fields = {k: v for k, v in (p.split("=", 1) for p in parts[1:])}
            key = (fields["defense"], fields["attack"], fields["scale"], fields["seed"])
            runs[key] = fields["run_id"]
    return runs


def server_accs(rid):
    accs = []
    with open(os.path.join(RESULTS, f"run_{rid}_server.csv")) as f:
        for r in csv.DictReader(f):
            accs.append(float(r["server_acc"]))
    return accs


def detection(rid):
    tp = fp = fn = tn = 0
    with open(os.path.join(RESULTS, f"run_{rid}_train.csv")) as f:
        for r in csv.DictReader(f):
            a, e = int(r["is_attacker"]), int(r["excluded"])
            if a and e:
                tp += 1
            elif a and not e:
                fn += 1
            elif not a and e:
                fp += 1
            else:
                tn += 1
    tpr = tp / (tp + fn) if (tp + fn) else None
    fpr = fp / (fp + tn) if (fp + tn) else None
    return tpr, fpr, tp + fn, fp + tn


def fmt(xs, nd=3):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "—"
    if len(xs) == 1:
        return f"{xs[0]:.4f}"
    return f"{statistics.mean(xs):.{nd}f}±{statistics.stdev(xs):.{nd}f}"


def main():
    runs = load_runs()
    print(f"Loaded {len(runs)} run cells from {SUMMARY}")

    print("\n" + "=" * 78)
    print("TABLE 1: Final accuracy (round 50) — mean ± std across 3 seeds")
    print("=" * 78)
    hdr = "defense        " + "".join(f"{a:>14}" for a in CELLS)
    print(hdr)
    for d in DEFENSES:
        row = f"{d:15s}"
        for cell, (atk, scl) in CELLS.items():
            accs = []
            for s in ("0", "1", "2"):
                rid = runs.get((d, atk, scl, s))
                if rid:
                    accs.append(server_accs(rid)[-1])
            row += f"{fmt(accs):>14}"
        print(row)

    print("\n" + "=" * 78)
    print("TABLE 1b: Mean of last-5-round accuracy — mean ± std across 3 seeds")
    print("=" * 78)
    print(hdr)
    for d in DEFENSES:
        row = f"{d:15s}"
        for cell, (atk, scl) in CELLS.items():
            accs = []
            for s in ("0", "1", "2"):
                rid = runs.get((d, atk, scl, s))
                if rid:
                    accs.append(sum(server_accs(rid)[-5:]) / 5)
            row += f"{fmt(accs):>14}"
        print(row)

    print("\n" + "=" * 78)
    print("TABLE 2: Detection — TPR / FPR (mean ± std across 3 seeds)")
    print("=" * 78)
    for d in ["mad", "krum"]:
        for cell in ["s2", "s5", "noise"]:
            atk, scl = CELLS[cell]
            tprs, fprs = [], []
            for s in ("0", "1", "2"):
                rid = runs.get((d, atk, scl, s))
                if not rid:
                    continue
                tpr, fpr, _, _ = detection(rid)
                if tpr is not None:
                    tprs.append(tpr)
                if fpr is not None:
                    fprs.append(fpr)
            print(f"  {d:6s} {cell:5s}  TPR={fmt(tprs)}  FPR={fmt(fprs)}")

    print("\nClean-run excluded flags (FPR among honest clients; no attacker present):")
    for d in DEFENSES:
        fprs = []
        for s in ("0", "1", "2"):
            rid = runs.get((d, "clean", "none", s))
            if not rid:
                continue
            _, fpr, _, hon = detection(rid)
            if fpr is not None:
                fprs.append(fpr)
        print(f"  {d:13s} FPR={fmt(fprs)}")

    print("\n" + "=" * 78)
    print("TABLE 3: Federated wall-clock (seconds) — mean ± std per defense")
    print("=" * 78)
    cell_of = {v: k for k, v in CELLS.items()}
    wc = collections.defaultdict(list)
    for (d, atk, scl, s), rid in runs.items():
        cell = cell_of.get((atk, scl))
        if cell is None:
            continue
        log = os.path.join(RESULTS, "logs", f"pillar_a_{cell}_{d}_seed{s}.log")
        if not os.path.exists(log):
            continue
        txt = open(log, errors="ignore").read()
        m = re.search(r"Strategy execution finished in ([\d.]+)s", txt)
        if m:
            wc[d].append(float(m.group(1)))
    for d in DEFENSES:
        v = wc[d]
        if v:
            print(f"  {d:13s} {statistics.mean(v):7.1f} ± {statistics.stdev(v):5.1f}  (n={len(v)})")

    print("\n" + "=" * 78)
    print("TABLE 4: registry dedup check")
    print("=" * 78)
    seen, dup, total = set(), 0, 0
    for line in open(os.path.join(RESULTS, "run_registry.jsonl")):
        try:
            e = json.loads(line)
        except Exception:
            continue
        total += 1
        if e["run_id"] in seen:
            dup += 1
        else:
            seen.add(e["run_id"])
    print(f"  registry lines={total}, unique run_ids={len(seen)}, duplicate lines={dup}")


if __name__ == "__main__":
    main()
