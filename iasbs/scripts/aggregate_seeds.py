#!/usr/bin/env python
"""Collapse the per-seed result JSONs written by multiseed.sh into mean +- std.

multiseed.sh writes one artifact per seed, ``json/results_<base>_s<S>.json``,
because none of the four discrete mains loops over seeds internally and
``save_ckpt`` has no seed suffix.  This script reads those files back, plus the
original single-seed ``json/results_<base>.json`` when it exists, flattens every
numeric leaf outside ``config``/``history`` into a dot path, and reports the
across-seed mean, sample standard deviation and per-seed values for every path
all the seeds agree on.

    python iasbs/scripts/aggregate_seeds.py occ_s32 ising_nd_L4
    python iasbs/scripts/aggregate_seeds.py --json json/seed_summary.json occ_s32

For the GB1 chain pass the stage-C stem, e.g. ``fs_gb1_k3_dirac`` with
``--stage C``; the script then looks for ``fs_gb1_k3_dirac_s<S>_C.json``.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys

SKIP_TOP = {"config", "history", "mult_err"}


def flatten(obj, prefix="", out=None):
    """Numeric leaves of a nested dict, keyed by dot path."""
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not prefix and k in SKIP_TOP:
                continue
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)) and math.isfinite(obj):
        out[prefix] = float(obj)
    return out


def seed_files(base, stage=None):
    """(seed, path) pairs for one base tag, sorted by seed."""
    suffix = f"_{stage}" if stage else ""
    hits = {}
    # ``_s<N>`` is what multiseed.sh writes; ``_seed<N>`` is the older hand-made
    # naming that the occupation m=32 non-Dirac pair already uses.
    for p in glob.glob(f"json/results_{base}_s*{suffix}.json"):
        m = re.search(rf"_(?:s|seed)(\d+){re.escape(suffix)}\.json$", p)
        if m:
            hits.setdefault(int(m.group(1)), p)
    # The pre-multiseed run is seed 0 by construction; keep it only if the
    # multiseed rerun of seed 0 is absent, so we never double count.
    orig = f"json/results_{base}{suffix}.json"
    if os.path.exists(orig) and 0 not in hits:
        hits[0] = orig
    return sorted(hits.items())


def summarise(base, stage=None):
    files = seed_files(base, stage)
    if not files:
        return None
    per_seed = {}
    for s, p in files:
        with open(p) as fh:
            per_seed[s] = flatten(json.load(fh))
    common = set.intersection(*(set(v) for v in per_seed.values()))
    seeds = [s for s, _ in files]
    rows = {}
    for path in sorted(common):
        vals = [per_seed[s][path] for s in seeds]
        n = len(vals)
        mean = sum(vals) / n
        std = (
            math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else 0.0
        )
        rows[path] = {"mean": mean, "std": std, "n": n,
                      "per_seed": dict(zip(seeds, vals))}
    return {"base": base, "stage": stage, "seeds": seeds,
            "files": dict(files), "metrics": rows}


def fmt(x):
    if x == 0:
        return "0"
    a = abs(x)
    return f"{x:.6g}" if 1e-4 <= a < 1e6 else f"{x:.4e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bases", nargs="+")
    ap.add_argument("--stage", default=None,
                    help="annealed-chain stage suffix, e.g. C for GB1")
    ap.add_argument("--only", default=None,
                    help="comma-separated substrings; keep matching metric paths")
    ap.add_argument("--json", default=None, help="also write the summary here")
    args = ap.parse_args()

    keep = [s.strip() for s in args.only.split(",")] if args.only else None
    allout = {}
    for base in args.bases:
        s = summarise(base, args.stage)
        if s is None:
            print(f"{base}: no seed files found", file=sys.stderr)
            continue
        allout[base] = s
        print(f"\n=== {base}" + (f" (stage {args.stage})" if args.stage else ""))
        print(f"    seeds {s['seeds']}  ->  " +
              ", ".join(os.path.basename(p) for p in s["files"].values()))
        for path, r in s["metrics"].items():
            if keep and not any(k in path for k in keep):
                continue
            vals = " ".join(f"s{k}={fmt(v)}" for k, v in r["per_seed"].items())
            print(f"    {path:34s} {fmt(r['mean']):>12s} +- {fmt(r['std']):<11s} | {vals}")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(allout, fh, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
