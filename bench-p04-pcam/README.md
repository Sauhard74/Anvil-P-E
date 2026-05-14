# Anvil P-04 · PCAM Precision Agent — Benchmark Harness

Reference benchmark for **P-04 · Precision-Controlled Associative Memory**.

Built on the PCAM paper (NeurIPS 2026 submission). The base PCAM model is provided to you, frozen — your job is to design an agent that picks a precision vector for each corrupted query so the system retrieves the correct stored pattern.

## Quickstart

```bash
cd bench-p04-pcam
pip install -r requirements.txt
python self_check.py --adapter adapters.dummy:DummyAgent --quick
```

This runs the identity-precision baseline (Π=I) so you can see the harness work end-to-end before writing any code.

## What you implement

Copy `adapters/dummy.py` to `adapters/myteam.py` and replace the `predict_precision` body:

```python
from adapter import Adapter
import numpy as np

class Engine(Adapter):
    def __init__(self, stored_patterns, model_params):
        """
        stored_patterns: (K, N) — patterns already stored in the system
        model_params:    dict with R, eta, beta, dt, T_max, tol, pi_min, pi_max
        """
        self.X = stored_patterns
        self.N = stored_patterns.shape[1]
        # one-time prep here — train a model, compute statistics, etc.

    def predict_precision(self, corrupted_query):
        """
        corrupted_query: (N,) noisy input
        returns:         (N,) positive precision values
        """
        # your logic
        return np.ones(self.N)
```

Then run:

```bash
python self_check.py --adapter adapters.myteam:Engine --quick
python run.py        --adapter adapters.myteam:Engine --out report.json
```

## Layout

```
pcam_model.py     Frozen PCAM dynamics (energy, gradient, Hessian, integrator)
data.py           Synthetic pattern generation + corruption
adapter.py        Adapter abstract base class
checks.py         Retrieval accuracy + anisotropy spread checks
run.py            Full evaluation CLI
self_check.py     Condensed CLI for local iteration
adapters/
  dummy.py        Π=I baseline (no precision modulation)
```

## What gets judged

| Check                | Weight | Definition                                                                |
|----------------------|--------|---------------------------------------------------------------------------|
| Retrieval Accuracy   | 70%    | Δ accuracy over Π=I baseline across 4 noise levels × 250 queries          |
| Anisotropy Spread    | 20%    | Spread reduction factor on 20 sampled stored-pattern attractors           |
| Code Quality         | 10%    | Manual — working code, reproducibility, README, design notes              |

### Retrieval scoring

- `Δ ≤ 0`: zero on this axis (you didn't beat the baseline)
- `Δ ≥ 0.05`: full marks (70 pts)
- Linear in between

### Anisotropy scoring

- Factor `≤ 1×`: zero
- Factor `5×`: half marks (10 pts)
- Factor `≥ 10×`: full marks (20 pts)
- Log-scaled in between
- (The paper achieves ~30× with an explicitly aligned construction; ~10× is a strong submission)

## Design hints

- **Variance-based**: down-weight dimensions whose value in the query looks like noise.
- **Class-conditional**: predict the class first (e.g., nearest stored pattern under cosine similarity), then set precision to match that class's typical pattern.
- **Geometry-aware**: read the local Hessian `model.hessian(approx_attractor)` and pick precision values that isotropise the spectrum of `ΠH` — this is what produces the ~30× spread reduction in the paper (Theorem F3).
- **Neural**: train a small MLP on (corrupted query, good precision) pairs you generate from the stored patterns.

Mix freely.

## Constraints

- The PCAM model is **frozen**. You don't modify `pcam_model.py`.
- Precision must be **diagonal and positive**. The harness clips to `[0.1, 10.0]` and mean-normalises to 1 before applying.
- One **forward pass** per query — no iterative refinement after observing the dynamics.

## v0 notes

This is the public iteration bench. It uses **synthetic random patterns** and **additive Gaussian corruption** in 64-dim space, matching the paper's synthetic-experiment setup (Section 6.3). It runs on CPU in well under 10 minutes per submission.

The final held-out evaluation will swap in **PCA-MNIST** with **mask noise** (the paper's Section 6.6 setup). Same harness, same interface, different data. Your agent's design should generalise.

## Hardware

NumPy only. CPU only. No GPU required.
