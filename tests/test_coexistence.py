import numpy as np
import pytest

from qcs.coexistence import (
    CoexistenceConfig,
    Propagation,
    apply_coexistence_noise,
    classical_load_profile,
    classical_throughput_factor,
)


@pytest.mark.parametrize("kind", ["diurnal", "bursty", "constant"])
def test_load_profile_bounded(kind):
    load = classical_load_profile(200, kind=kind, seed=0)
    assert load.shape == (200,)
    assert np.all(load >= 0.0) and np.all(load <= 1.0)


def test_unknown_load_kind_raises():
    with pytest.raises(ValueError):
        classical_load_profile(10, kind="nonsense")


def test_noise_clipped_to_unit_interval():
    cfg = CoexistenceConfig(propagation=Propagation.CO, co_noise_coeff=5.0)
    w = apply_coexistence_noise(clean_w=0.99, classical_load=1.0, cfg=cfg)
    assert 0.0 <= w <= 1.0


def test_counter_propagation_less_noisy_than_co():
    cfg_co = CoexistenceConfig(propagation=Propagation.CO)
    cfg_counter = CoexistenceConfig(propagation=Propagation.COUNTER)
    w_co = apply_coexistence_noise(0.99, 0.8, cfg_co)
    w_counter = apply_coexistence_noise(0.99, 0.8, cfg_counter)
    assert w_counter > w_co


def test_interleaved_has_least_noise_but_costs_classical_throughput():
    cfg_interleaved = CoexistenceConfig(propagation=Propagation.INTERLEAVED)
    cfg_co = CoexistenceConfig(propagation=Propagation.CO)
    w_interleaved = apply_coexistence_noise(0.99, 0.8, cfg_interleaved)
    w_co = apply_coexistence_noise(0.99, 0.8, cfg_co)
    assert w_interleaved > w_co

    assert classical_throughput_factor(Propagation.INTERLEAVED, cfg_interleaved) < 1.0
    assert classical_throughput_factor(Propagation.CO, cfg_co) == 1.0


def test_zero_load_leaves_fidelity_unchanged():
    cfg = CoexistenceConfig(propagation=Propagation.CO)
    assert apply_coexistence_noise(0.9, 0.0, cfg) == pytest.approx(0.9)
