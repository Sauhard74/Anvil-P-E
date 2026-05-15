"""
ANVIL · P-04 · L3 Final Benchmark Runner

This is the ONLY bench for P-04. Running this script IS the L3 evaluation.
The output JSON is what participants paste into the submission form.

Usage:
    python run.py --adapter adapters.myteam:Engine --out l3_report.json
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
import time
from typing import Any, Callable

import numpy as np

from harness import run_multi


L3_VERSION = "anvil-2026-p04-L3-final"


# =====================================================================
# COUNCIL: Replace these seeds at T-2h with the L3 release values.
#
# Why this matters: the bench gives the agent the stored-pattern
# matrix X at __init__. With public, predictable seeds, an attacker
# can hash X to identify the seed, regenerate the exact corrupted-
# query test set the harness will use, and build a {q → correct idx}
# lookup table. Seed rotation kills this attack: a precomputation
# against the public seeds becomes useless the moment seeds change.
# =====================================================================
L3_SEEDS = [42, 101, 202, 303, 404]
L3_K = 16
L3_N = 64
L3_NOISE_LEVELS = [0.6, 0.75, 0.85]
L3_N_PER_LEVEL = 250
L3_N_ANISOTROPY = 16
# =====================================================================


# --------------------------------------------------------------------------- #
# Visual banners — designed for video identification.
# --------------------------------------------------------------------------- #

def _banner_open() -> str:
    bar = "█" * 70
    star = "★" * 3
    return "\n".join([
        "",
        bar,
        bar,
        f"{star}     A N V I L   ·   P - 0 4   ·   L 3   F I N A L   B E N C H     {star}",
        f"{star}     Council Release · {L3_VERSION:<32}     {star}",
        f"{star}     {time.strftime('%Y-%m-%d %H:%M:%S %z'):<58}{star}",
        bar,
        bar,
        "",
    ])


def _banner_close(score_value: float, score_max: float) -> str:
    bar = "█" * 70
    star = "★" * 3
    pct = (score_value / score_max * 100) if score_max else 0.0
    return "\n".join([
        "",
        bar,
        bar,
        f"{star}     A N V I L   ·   P - 0 4   ·   L 3   F I N A L   S C O R E     {star}",
        f"{star}     {score_value:>6.4f}  /  {score_max:.4f}    ({pct:>5.1f} %)         {star}",
        f"{star}     {L3_VERSION:<58}{star}",
        bar,
        bar,
        "",
    ])


def agent_factory_from_spec(spec: str) -> tuple[Callable[[np.ndarray, dict[str, Any]], Any], str, str]:
    """Returns (factory, source_path, source_sha256)."""
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    try:
        source_path = module.__file__ or "<unknown>"
        with open(source_path, "rb") as f:
            source = f.read()
        source_hash = hashlib.sha256(source).hexdigest()
    except Exception:
        source_path = "<unknown>"
        source_hash = "<unhashable>"
    def factory(X: np.ndarray, params: dict[str, Any]):
        return cls(X, params)
    return factory, source_path, source_hash


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Anvil · P-04 · L3 final benchmark (single mode, single output)",
    )
    ap.add_argument("--adapter", required=True,
                    help="module:Class, e.g. adapters.myteam:Engine")
    ap.add_argument("--seeds", type=int, nargs="+", default=L3_SEEDS,
                    help="L3 seeds. Council rotates these at T-2h.")
    ap.add_argument("--K", type=int, default=L3_K)
    ap.add_argument("--N", type=int, default=L3_N)
    ap.add_argument("--noise-levels", type=float, nargs="+", default=L3_NOISE_LEVELS)
    ap.add_argument("--n-per-level", type=int, default=L3_N_PER_LEVEL)
    ap.add_argument("--n-anisotropy", type=int, default=L3_N_ANISOTROPY)
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)

    # --- OPEN BANNER ---
    sys.stderr.write(_banner_open())
    sys.stderr.write(
        f"  ▸ Adapter:           {args.adapter}\n"
        f"  ▸ Seeds:             {args.seeds}\n"
        f"  ▸ K (patterns):      {args.K}\n"
        f"  ▸ N (dimensions):    {args.N}\n"
        f"  ▸ Noise levels:      {args.noise_levels}\n"
        f"  ▸ Queries per level: {args.n_per_level}\n\n"
    )
    sys.stderr.flush()

    factory, adapter_path, adapter_hash = agent_factory_from_spec(args.adapter)
    sys.stderr.write(
        f"  ▸ Adapter source:    {adapter_path}\n"
        f"  ▸ Adapter SHA-256:   {adapter_hash[:16]}…\n\n"
        "  ▸ Running L3 evaluation across all seeds …\n"
    )
    sys.stderr.flush()

    report = run_multi(
        agent_factory=factory,
        seeds=args.seeds,
        K=args.K, N=args.N,
        noise_levels=args.noise_levels,
        n_per_level=args.n_per_level,
        n_aniso=args.n_anisotropy,
    )

    report["l3_version"] = L3_VERSION
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    report["adapter"] = args.adapter
    report["adapter_path"] = adapter_path
    report["adapter_sha256"] = adapter_hash
    report["seeds"] = args.seeds

    sc = report["score"]
    final = sc["total_automated"]
    final_max = sc.get("max_automated", 90.0)

    payload = json.dumps(report, indent=2, default=str)
    if args.out == "-":
        print(payload)
    else:
        with open(args.out, "w") as f:
            f.write(payload)

    # --- CLOSE BANNER ---
    sys.stderr.write(_banner_close(final, final_max))
    sys.stderr.flush()

    return 0


if __name__ == "__main__":
    sys.exit(main())
