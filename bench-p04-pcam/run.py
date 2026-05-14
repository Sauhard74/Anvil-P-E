"""
P-04 benchmark runner.

Usage:
    python run.py --adapter adapters.dummy:DummyAgent
    python run.py --adapter adapters.myteam:Engine --out report.json
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time

import numpy as np

from adapters.dummy import DummyAgent
from checks import retrieval_accuracy, spread_reduction
from data import make_patterns, make_test_queries
from pcam_model import PCAMModel, build_default_R


def load_adapter(spec: str):
    module_name, class_name = spec.split(":")
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _retrieval_score(delta: float, full_at: float = 0.05, weight: float = 70.0) -> float:
    if delta <= 0:
        return 0.0
    return float(min(weight, weight * (delta / full_at)))


def _spread_score(factor: float, half_at: float = 5.0, full_at: float = 10.0,
                  weight: float = 20.0) -> float:
    if factor <= 1.0:
        return 0.0
    return float(min(weight, weight * (np.log(factor) / np.log(full_at))))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Anvil P-04 · PCAM benchmark")
    ap.add_argument("--adapter", required=True,
                    help="module:Class, e.g. adapters.myteam:Engine")
    ap.add_argument("--K", type=int, default=16, help="Stored patterns (v0 synthetic default matches paper Section 6.1)")
    ap.add_argument("--N", type=int, default=64, help="State dimension")
    ap.add_argument("--noise-levels", type=float, nargs="+",
                    default=[0.5, 0.7, 0.8])
    ap.add_argument("--n-per-level", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-anisotropy", type=int, default=16)
    ap.add_argument("--out", default="-")
    args = ap.parse_args(argv)

    print(f"[{time.strftime('%H:%M:%S')}] building PCAM model ...", file=sys.stderr)
    X = make_patterns(K=args.K, N=args.N, seed=42)
    R = build_default_R(N=args.N, seed=42)
    model = PCAMModel(X, R)
    params = {
        "R": R, "eta": model.eta, "beta": model.beta,
        "dt": model.dt, "T_max": model.T_max, "tol": model.tol,
        "pi_min": model.pi_min, "pi_max": model.pi_max,
    }

    print(f"[{time.strftime('%H:%M:%S')}] generating test queries ...", file=sys.stderr)
    queries, truths, levels = make_test_queries(
        X, args.noise_levels, args.n_per_level, args.seed,
    )

    AgentClass = load_adapter(args.adapter)
    agent = AgentClass(X, params)
    dummy = DummyAgent(X, params)

    print(f"[{time.strftime('%H:%M:%S')}] running retrieval check "
          f"({len(queries)} queries) ...", file=sys.stderr)
    t0 = time.monotonic()
    base_acc = retrieval_accuracy(model, dummy, queries, truths)
    base_t = time.monotonic() - t0
    t0 = time.monotonic()
    agent_acc = retrieval_accuracy(model, agent, queries, truths)
    agent_t = time.monotonic() - t0
    delta = agent_acc - base_acc

    print(f"[{time.strftime('%H:%M:%S')}] running anisotropy check "
          f"({args.n_anisotropy} patterns) ...", file=sys.stderr)
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(args.K, size=args.n_anisotropy, replace=False).tolist()
    t0 = time.monotonic()
    spread = spread_reduction(model, agent, dummy, indices, seed=args.seed)
    spread_t = time.monotonic() - t0

    retrieval_pts = _retrieval_score(delta)
    spread_pts = _spread_score(spread["reduction_factor"])

    report = {
        "config": {
            "K": args.K, "N": args.N,
            "noise_levels": args.noise_levels,
            "n_per_level": args.n_per_level,
            "seed": args.seed,
            "n_anisotropy": args.n_anisotropy,
        },
        "retrieval": {
            "agent_accuracy":    round(agent_acc, 4),
            "baseline_accuracy": round(base_acc, 4),
            "delta":             round(delta, 4),
            "agent_time_s":      round(agent_t, 2),
            "baseline_time_s":   round(base_t, 2),
        },
        "anisotropy": {
            **spread,
            "time_s": round(spread_t, 2),
            "n_patterns": args.n_anisotropy,
        },
        "score": {
            "retrieval_pts":     round(retrieval_pts, 2),
            "anisotropy_pts":    round(spread_pts, 2),
            "code_quality_pts":  "(manual, up to 10)",
            "total_automated":   round(retrieval_pts + spread_pts, 2),
            "max_automated":     90.0,
        },
    }

    payload = json.dumps(report, indent=2, default=str)
    if args.out == "-":
        print(payload)
    else:
        with open(args.out, "w") as f:
            f.write(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
