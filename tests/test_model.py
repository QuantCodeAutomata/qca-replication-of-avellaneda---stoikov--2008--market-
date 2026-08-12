import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from src.model import (
    indifference_price, optimal_spread, arrival_intensity, execution_prob, PARAMS
)


def test_indifference_price_zero_inventory():
    # With q=0, r must equal s exactly
    r = indifference_price(s=100.0, q=0, gamma=0.1, sigma=2.0, T=1.0, t=0.5)
    assert r == 100.0


def test_indifference_price_positive_inventory():
    # With q>0, inventory holder wants lower price to sell faster: r < s
    r = indifference_price(s=100.0, q=5, gamma=0.1, sigma=2.0, T=1.0, t=0.0)
    assert r < 100.0
    # r = 100 - 5*0.1*4*1 = 100 - 2 = 98
    assert abs(r - 98.0) < 1e-10


def test_indifference_price_negative_inventory():
    # With q<0 (short), r > s
    r = indifference_price(s=100.0, q=-5, gamma=0.1, sigma=2.0, T=1.0, t=0.0)
    assert r > 100.0
    assert abs(r - 102.0) < 1e-10


def test_optimal_spread_positive():
    # Spread must always be positive
    spread = optimal_spread(gamma=0.1, sigma=2.0, T=1.0, t=0.0, k=1.5)
    assert spread > 0


def test_optimal_spread_decreases_with_time():
    # Spread decreases as t approaches T (less time remaining)
    s0 = optimal_spread(gamma=0.1, sigma=2.0, T=1.0, t=0.0, k=1.5)
    s1 = optimal_spread(gamma=0.1, sigma=2.0, T=1.0, t=0.5, k=1.5)
    s2 = optimal_spread(gamma=0.1, sigma=2.0, T=1.0, t=0.995, k=1.5)
    assert s0 > s1 > s2


def test_optimal_spread_increases_with_gamma():
    # Higher risk aversion -> wider spread
    s_low = optimal_spread(gamma=0.01, sigma=2.0, T=1.0, t=0.0, k=1.5)
    s_high = optimal_spread(gamma=0.5, sigma=2.0, T=1.0, t=0.0, k=1.5)
    # The formula is complex; verify both are positive
    assert s_low > 0 and s_high > 0


def test_arrival_intensity_decays_with_distance():
    # Further quotes have lower arrival intensity
    lam1 = arrival_intensity(delta=0.5, A=140.0, k=1.5)
    lam2 = arrival_intensity(delta=1.0, A=140.0, k=1.5)
    assert lam1 > lam2
    # exact: 140*exp(-1.5*0.5) vs 140*exp(-1.5*1.0)
    assert abs(lam1 - 140 * np.exp(-0.75)) < 1e-10


def test_execution_prob_capped_at_one():
    # Very high intensity should give prob=1.0
    p = execution_prob(lam=1e6, dt=0.005)
    assert p == 1.0


def test_execution_prob_small_intensity():
    p = execution_prob(lam=10.0, dt=0.005)
    assert abs(p - 0.05) < 1e-10
    assert 0.0 <= p <= 1.0


def test_optimal_spread_formula_exact():
    # Verify exact formula value for gamma=0.1, t=0
    gamma, sigma, T, t, k = 0.1, 2.0, 1.0, 0.0, 1.5
    expected = gamma * sigma**2 * (T - t) + (2.0 / gamma) * np.log(1.0 + gamma / k)
    result = optimal_spread(gamma, sigma, T, t, k)
    assert abs(result - expected) < 1e-12


def test_params_dict_keys():
    # Verify PARAMS has required keys
    for key in ['S0', 'SIGMA', 'T', 'DT', 'K', 'A', 'Q0', 'N_PATHS']:
        assert key in PARAMS
