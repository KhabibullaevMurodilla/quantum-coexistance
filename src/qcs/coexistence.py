"""
coexistence.py
===============
Classical-traffic-driven noise on a shared quantum-classical fiber.

When a quantum channel shares fiber with classical DWDM traffic, spontaneous
Raman scattering from the classical signal injects extra photons into the
quantum channel's band, degrading the delivered Werner parameter. This is
the standard, physically-motivated mechanism in the quantum-classical
coexistence literature (not specific to any one paper): co-propagating
classical traffic causes substantially more noise than counter-propagating
traffic, because counter-propagating Raman-scattered photons have to
traverse the full fiber length against the signal direction and are far
more strongly attenuated before reaching the receiver.

This module turns a classical traffic *load* time series into an extra
depolarization term on top of whatever "clean" Werner parameter a link
would deliver, so a scheduler has to trade classical throughput against
quantum fidelity rather than treating them as independent resources.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class Propagation(Enum):
    CO = "co"        # classical and quantum signals travel the same direction
    COUNTER = "counter"  # opposite directions -- much less Raman noise reaches the receiver
    INTERLEAVED = "interleaved"  # time-division: classical and quantum alternate in separate slots


@dataclass
class CoexistenceConfig:
    propagation: Propagation = Propagation.COUNTER
    # Noise coefficients are illustrative, order-of-magnitude figures
    # consistent with the qualitative co- vs. counter-propagating gap
    # reported in the quantum-classical coexistence literature (roughly an
    # order of magnitude worse for co-propagation) -- not fit to any single
    # paper's measured values. Treat as a tunable knob, not ground truth.
    co_noise_coeff: float = 0.35
    counter_noise_coeff: float = 0.04
    # Fraction of classical throughput sacrificed when a link is run in
    # interleaved (time-division) mode; interleaving removes Raman noise
    # almost entirely but at a direct cost to classical capacity.
    interleave_classical_penalty: float = 0.5
    interleave_residual_noise_coeff: float = 0.005


def classical_load_profile(n_steps: int, kind: str = "diurnal", seed: int = 0) -> np.ndarray:
    """A per-timestep classical traffic load in [0, 1] (fraction of a link's
    classical capacity in use). `kind`:
      - "diurnal": smooth day/night sinusoid + noise, representative of real
        backbone traffic engineering traces.
      - "bursty": on/off bursts (heavy-tailed idle/active periods).
      - "constant": fixed 50% load, useful as a sanity-check baseline.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    if kind == "constant":
        return np.full(n_steps, 0.5)

    if kind == "diurnal":
        base = 0.5 + 0.35 * np.sin(2 * np.pi * t / max(1, n_steps / 3) - np.pi / 2)
        noise = rng.normal(0, 0.05, n_steps)
        return np.clip(base + noise, 0.02, 0.98)

    if kind == "bursty":
        load = np.zeros(n_steps)
        i = 0
        state = rng.random() < 0.3
        while i < n_steps:
            run = max(1, int(rng.exponential(8 if state else 4)))
            load[i:i + run] = (0.75 + 0.2 * rng.random()) if state else (0.05 + 0.1 * rng.random())
            i += run
            state = not state
        return np.clip(load, 0.0, 1.0)

    raise ValueError(f"Unknown load profile kind: {kind!r}")


def apply_coexistence_noise(clean_w: float, classical_load: float, cfg: CoexistenceConfig) -> float:
    """Degrade a link's clean Werner parameter under the given classical
    load and propagation scheme. Returns the effective w seen by the
    scheduler, clipped to [0, 1].
    """
    if cfg.propagation == Propagation.INTERLEAVED:
        extra_depol = cfg.interleave_residual_noise_coeff * classical_load
    elif cfg.propagation == Propagation.CO:
        extra_depol = cfg.co_noise_coeff * classical_load
    else:  # COUNTER
        extra_depol = cfg.counter_noise_coeff * classical_load

    # Depolarizing composition: an extra depolarizing channel with
    # parameter (1 - extra_depol) composed with the clean Werner state
    # multiplies the Werner parameter (same algebra as end_to_end_fidelity's
    # composition rule in network.py -- depolarizing noise composes
    # multiplicatively on the Werner parameter).
    w_eff = clean_w * max(0.0, 1.0 - extra_depol)
    return float(np.clip(w_eff, 0.0, 1.0))


def classical_throughput_factor(propagation: Propagation, cfg: CoexistenceConfig) -> float:
    """What fraction of a link's nominal classical capacity remains
    available under this propagation scheme -- interleaving buys lower
    quantum noise at a direct classical-throughput cost; co/counter
    propagation cost nothing in classical throughput."""
    if propagation == Propagation.INTERLEAVED:
        return 1.0 - cfg.interleave_classical_penalty
    return 1.0
