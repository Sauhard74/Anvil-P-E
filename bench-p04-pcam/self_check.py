"""
P-04 self-check.

Quick condensed run for local iteration. Prints a friendly summary
instead of dumping JSON.

    python self_check.py --adapter adapters.dummy:DummyAgent --quick
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

import numpy as np

from adapters.dummy import DummyAgent
from checks import retrieval_accuracy, spread_reduction
from data import make_patterns, make_test_queries
from pcam_model import PCAMModel, build_default_R


def load_adapter(spec: str):
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P-04 self-check")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="Smaller K, fewer queries — fast iteration.")
    args = ap.parse_args(argv)

    if args.quick:
        K, N = 16, 64
        noise_levels = [0.7, 0.8]
        n_per_level = 50
        n_aniso = 5
    else:
        K, N = 16, 64
        noise_levels = [0.5, 0.7, 0.8]
        n_per_level = 250
        n_aniso = 16

    X = make_patterns(K=K, N=N, seed=42)
    R = build_default_R(N=N, seed=42)
    model = PCAMModel(X, R)
    params = {
        "R": R, "eta": model.eta, "beta": model.beta,
        "dt": model.dt, "T_max": model.T_max, "tol": model.tol,
        "pi_min": model.pi_min, "pi_max": model.pi_max,
    }

    queries, truths, _ = make_test_queries(X, noise_levels, n_per_level, seed=0)

    AgentClass = load_adapter(args.adapter)
    agent = AgentClass(X, params)
    dummy = DummyAgent(X, params)

    t0 = time.monotonic()
    base_acc = retrieval_accuracy(model, dummy, queries, truths)
    agent_acc = retrieval_accuracy(model, agent, queries, truths)
    rng = np.random.default_rng(0)
    indices = rng.choice(K, size=n_aniso, replace=False).tolist()
    spread = spread_reduction(model, agent, dummy, indices, seed=0)
    total_ms = (time.monotonic() - t0) * 1000.0

    delta = agent_acc - base_acc
    factor = spread["reduction_factor"]

    retrieval_pts = 0.0 if delta <= 0 else min(70.0, 70.0 * (delta / 0.05))
    spread_pts = 0.0 if factor <= 1.0 else min(20.0, 20.0 * (np.log(factor) / np.log(10.0)))

    print()
    print("ANVIL · P-04 · PCAM Precision Agent — Self-Check")
    print("=" * 60)
    print(f"  total wall time          {total_ms:>10.1f} ms")
    print(f"  stored patterns (K)      {K:>10d}")
    print(f"  state dim (N)            {N:>10d}")
    print(f"  test queries             {len(queries):>10d}")
    print(f"  anisotropy samples       {n_aniso:>10d}")
    print()
    print("  CHECK                                VALUE")
    print("  " + "-" * 50)
    print(f"  retrieval acc — agent              {agent_acc:>6.3f}")
    print(f"  retrieval acc — Π=I baseline       {base_acc:>6.3f}")
    print(f"  Δ accuracy (you - baseline)        {delta:+.3f}")
    print(f"  anisotropy spread reduction        {factor:>6.2f}×")
    print()
    print("  SCORE (automated, max 90)           POINTS")
    print("  " + "-" * 50)
    print(f"  retrieval     (max 70)             {retrieval_pts:>6.2f}")
    print(f"  anisotropy    (max 20)             {spread_pts:>6.2f}")
    print(f"  code quality  (max 10)             (manual)")
    print(f"  TOTAL AUTOMATED                    {retrieval_pts + spread_pts:>6.2f}  / 90")
    print()
    if delta <= 0:
        print("  ⚠  Agent does not beat Π=I baseline on retrieval — zero on that axis.")
    elif delta < 0.02:
        print("  ▸  Agent is above baseline but the gain is small. Aim for Δ ≥ 0.05 for full marks.")
    else:
        print("  ✓  Solid retrieval gain.")
    if factor < 2.0:
        print("  ▸  Anisotropy spread is close to baseline. Aim for ≥ 10× for full marks.")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
