import pytest

from qcs.utility import UtilityKind, UtilityWeights, coherence_gate, jains_fairness_index, session_utility


@pytest.mark.parametrize("kind", [UtilityKind.LINEAR, UtilityKind.LOG_RATE, UtilityKind.NEGATIVITY])
def test_utility_zero_below_fidelity_floor(kind):
    u = session_utility(kind, fidelity=0.5, rate_ebit_s=100, latency_s=0.0, fidelity_floor=0.9)
    assert u == 0.0


@pytest.mark.parametrize("kind", [UtilityKind.LINEAR, UtilityKind.LOG_RATE, UtilityKind.NEGATIVITY])
def test_utility_nonnegative_above_floor(kind):
    u = session_utility(kind, fidelity=0.95, rate_ebit_s=10, latency_s=0.0001, fidelity_floor=0.9)
    assert u >= 0.0


def test_utility_increases_with_rate_log():
    low = session_utility(UtilityKind.LOG_RATE, 0.95, 1, 0.0, 0.9)
    high = session_utility(UtilityKind.LOG_RATE, 0.95, 100, 0.0, 0.9)
    assert high > low


def test_linear_utility_penalizes_latency():
    weights = UtilityWeights(latency_ref_s=1.0)
    fast = session_utility(UtilityKind.LINEAR, 0.95, 10, 0.0, 0.9, weights)
    slow = session_utility(UtilityKind.LINEAR, 0.95, 10, 1.0, 0.9, weights)
    assert fast > slow


def test_coherence_gate_decays_over_time():
    assert coherence_gate(0.9, 0.0, 1e-3) == pytest.approx(1.0)
    decayed = coherence_gate(0.9, 5e-3, 1e-3)
    assert 0.0 < decayed < 0.1


def test_coherence_gate_zero_coherence_time():
    assert coherence_gate(0.9, 0.0, 0.0) == 1.0
    assert coherence_gate(0.9, 1e-9, 0.0) == 0.0


def test_jains_fairness_index_bounds():
    assert jains_fairness_index([]) == 1.0
    assert jains_fairness_index([5, 5, 5, 5]) == pytest.approx(1.0)
    assert jains_fairness_index([10, 0, 0, 0]) == pytest.approx(0.25)
    unfair = jains_fairness_index([1, 100])
    assert 0.0 < unfair < 1.0
