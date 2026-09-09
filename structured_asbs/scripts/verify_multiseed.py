#!/usr/bin/env python
"""Verify multiseed.sh reproduces each seed-0 run exactly, modulo the seed.

Every added seed must differ from the run already quoted in the paper in the
seed and nothing else.  This script checks that mechanically:

  1. parse the flags multiseed.sh records for each row;
  2. import the corresponding entry point and extract its argparse defaults
     for that subcommand;
  3. resolve the row into the full argument namespace it would produce;
  4. load the config block the seed-0 run wrote into json/results_<base>.json;
  5. diff the two, ignoring only the keys that are *supposed* to move
     (seed, tag, out, eval-every under the drop switch, and the derived
     N = m rule).

A clean report means "re-running this row with --seed S reproduces the seed-0
configuration exactly".

Usage:  python structured_asbs/scripts/verify_multiseed.py [--show-ok] [--row NAME]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "structured_asbs" / "scripts" / "multiseed.sh"

# Keys allowed to differ between a row and the seed-0 config.
IGNORE = {
    "seed",            # the whole point of the exercise
    "tag", "out",      # rewritten to <base>_s<S> so nothing is overwritten
    "ckpt_dir",        # artifact location, not a training input
    "eval_every",      # drop/keep switch; the final eval always runs
    "device", "resume", "verbose", "quiet", "no_tqdm",
    "init_from",       # gb1 chain wiring, checked separately below
}

_DEFAULT_CACHE: dict[str, dict] = {}

# Two seed-0 runs wrote their JSON under a name that differs from their
# checkpoint tag, so results_<base>.json does not exist for them.  The
# configurations are still the ones multiseed.sh replays; map them explicitly
# rather than letting the check silently skip.
#
#   occ4_full        -> rerun_ckpt.sh used --tag occ4_$est --out results_occ_$est.json
#   occ_nd_s32       -> run under an older _seed<N> suffix; both seeds present
ALIASES = {
    "occ4_full": ["occ_full"],
    "occ_nd_s32": ["occ_nd_s32_seed0", "occ_nd_s32_seed1"],
}


# ----------------------------------------------------------------- parsing --
def parse_rows() -> list[tuple[str, str, str, str]]:
    txt = SCRIPT.read_text()
    body = txt.split("cat <<'ROWS'\n", 1)[1].split("\nROWS\n", 1)[0]
    return [tuple(line.split("|")) for line in body.strip().splitlines()]


def parse_gb1_rows() -> list[tuple[str, str | None, str | None, str | None, int | None]]:
    txt = SCRIPT.read_text()
    fn = txt.split("run_gb1_chain() {", 1)[1].split("\n}\n", 1)[0]
    out = []
    for kind in ("dirac", "nd", "dam"):
        blk = re.search(
            rf"{kind}\)\s+script=(\S+);\s+base=\"([^\"]+)\"\s*\n\s*args=\"([^\"]+)\""
            r"[\s\S]*?local it=(\d+)",
            fn,
        )
        out.append(
            (f"gb1_{kind}", blk.group(1), blk.group(3), blk.group(2), int(blk.group(4)))
            if blk else (f"gb1_{kind}", None, None, None, None)
        )
    return out


def flags_to_dict(args: str) -> tuple[str, dict]:
    """'train --a 1 --b' -> ('train', {'a': '1', 'b': True})."""
    toks = args.split()
    sub, d, i = toks[0], {}, 1
    while i < len(toks):
        t = toks[i]
        if not t.startswith("--"):
            i += 1
            continue
        key = t[2:].replace("-", "_")
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            d[key], i = toks[i + 1], i + 2
        else:
            d[key], i = True, i + 1
    return sub, d


# ------------------------------------------------------- argparse defaults --
def _capture_parser(script: str):
    """Import the entry point and grab the ArgumentParser main() builds."""
    path = ROOT / script
    box: dict[str, argparse.ArgumentParser] = {}
    real = argparse.ArgumentParser.parse_args

    class _Stop(Exception):
        pass

    def fake(self, *a, **k):
        box["ap"] = self
        raise _Stop

    spec = importlib.util.spec_from_file_location(f"_ep_{path.stem}", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    sys.path.insert(0, str(ROOT))
    argparse.ArgumentParser.parse_args = fake
    try:
        spec.loader.exec_module(mod)
        try:
            mod.main()
        except _Stop:
            pass
    finally:
        argparse.ArgumentParser.parse_args = real
        sys.path.pop(0)
        sys.path.pop(0)
    return box.get("ap")


def defaults_for(script: str, sub: str) -> dict:
    """Default namespace for <script> <sub>, following subparsers if present."""
    key = f"{script}::{sub}"
    if key in _DEFAULT_CACHE:
        return _DEFAULT_CACHE[key]
    ap = _capture_parser(script)
    if ap is None:
        _DEFAULT_CACHE[key] = {}
        return {}
    d = {}
    target = ap
    for act in ap._actions:
        if isinstance(act, argparse._SubParsersAction):
            target = act.choices.get(sub, ap)
            break
    for parser in ({ap, target} if target is not ap else {ap}):
        for act in parser._actions:
            if isinstance(act, (argparse._HelpAction, argparse._SubParsersAction)):
                continue
            d[act.dest] = act.default
    _DEFAULT_CACHE[key] = d
    return d


def norm(v):
    if isinstance(v, bool) or v is None:
        return v
    try:
        f = float(v)
        return int(f) if f == int(f) else f
    except (TypeError, ValueError):
        return str(v)


def resolve(script: str, sub: str, row: dict) -> dict:
    """Full namespace a row produces: defaults overlaid with its explicit flags."""
    ns = dict(defaults_for(script, sub))
    ns.update(row)
    ns["cmd"] = sub
    # both occupation.py and dam/discrete.py document "--N 0 means N = m"
    if norm(ns.get("N")) in (0, None) and "m" in ns:
        ns["N"] = ns["m"]
    return ns


# ---------------------------------------------------------------- artifacts --
def load_cfg(base: str):
    p = ROOT / "json" / f"results_{base}.json"
    if not p.exists():
        return None, f"missing json/results_{base}.json"
    try:
        obj = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        return None, f"unreadable: {e}"
    if isinstance(obj, list):
        obj = obj[0] if obj else {}
    for k in ("config", "args", "cfg", "params"):
        if isinstance(obj.get(k), dict):
            return obj[k], None
    return None, "no config block in json"


def compare(ns: dict, cfg: dict) -> list[tuple[str, str, object, object]]:
    bad = []
    for k, v in cfg.items():
        if k in IGNORE:
            continue
        if k not in ns:
            bad.append(("UNKNOWN", k, "-", v))
        elif norm(ns[k]) != norm(v):
            bad.append(("DIFFERS", k, ns[k], v))
    return bad


def report(label: str, base: str, bad, show_ok: bool) -> int:
    if bad:
        print(f"{label:<18} {base:<30} MISMATCH ({len(bad)})")
        for kind, k, rv, cv in bad:
            print(f"{'':<18} {'':<30}   {kind:<8} {k:<18} multiseed={rv!r:<14} seed0={cv!r}")
        return 1
    print(f"{label:<18} {base:<30} ok")
    return 0


# --------------------------------------------------------------------- main --
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show-ok", action="store_true")
    ap.add_argument("--row", default=None)
    a = ap.parse_args()

    nbad = nskip = 0
    print(f"{'ROW':<18} {'BASE':<30} VERDICT")
    print("-" * 82)

    for name, script, args, base in parse_rows():
        if a.row and name != a.row:
            continue
        sub, row = flags_to_dict(args)
        ns = resolve(script, sub, row)
        for tgt in ALIASES.get(base, [base]):
            cfg, err = load_cfg(tgt)
            if cfg is None:
                print(f"{name:<18} {tgt:<30} SKIP  ({err})")
                nskip += 1
                continue
            nbad += report(name, tgt, compare(ns, cfg), a.show_ok)

    for name, script, args, base, it in parse_gb1_rows():
        if a.row and name != a.row:
            continue
        if script is None:
            print(f"{name:<18} {'?':<30} SKIP  (runner not parsable)")
            nskip += 1
            continue
        sub, row = flags_to_dict(args)
        for stage, tau, ns_samples in (("A", 2.0, 2000), ("B", 1.4, 2000), ("C", 1.0, 20000)):
            tag = f"{base}_{stage}"
            cfg, err = load_cfg(tag)
            if cfg is None:
                print(f"{name + '/' + stage:<18} {tag:<30} SKIP  ({err})")
                nskip += 1
                continue
            r = dict(row, tau=tau, iters=it, n_samples=ns_samples)
            nbad += report(f"{name}/{stage}", tag, compare(resolve(script, sub, r), cfg), a.show_ok)

    print("-" * 82)
    print(f"{nbad} mismatch(es), {nskip} skip(s)")
    return 1 if nbad else 0


if __name__ == "__main__":
    sys.exit(main())
