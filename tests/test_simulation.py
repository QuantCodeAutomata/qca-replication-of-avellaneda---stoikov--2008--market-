import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pytest
from src.simulation import run_simulation, run_all_gammas, compute_stats
from src.model import PARAMS


def test_run_simulation_output_shape():
    rng = np.random.default_rng(42)
    n_paths = 50
    results = run_simulation(gamma=0.1, n_paths=n_paths, rng=rng, params=PARAMS)
    assert len(results['inventory_pnl']) == n_paths
    assert len(results['symmetric_pnl']) == n_paths


def test_run_simulation_returns_floats():
    rng = np.random.default_rng(0)
    results = run_simulation(gamma=0.1, n_paths=10, rng=rng, params=PARAMS)
    assert results['inventory_pnl'].dtype in [np.float64, np.float32]


def test_inventory_strategy_lower_std_than_symmetric():
    # Key hypothesis: inventory std < symmetric std for gamma=0.1
    rng = np.random.default_rng(42)
    results = run_simulation(gamma=0.1, n_paths=1000, rng=rng, params=PARAMS)
    inv_std = np.std(results['inventory_pnl'])
    sym_std = np.std(results['symmetric_pnl'])
    assert inv_std < sym_std, f'Expected inv_std < sym_std, got {inv_std:.2f} vs {sym_std:.2f}'


def test_inventory_strategy_lower_terminal_inventory_std():
    # Inventory strategy should also have lower terminal inventory variance
    rng = np.random.default_rng(42)
    results = run_simulation(gamma=0.1, n_paths=1000, rng=rng, params=PARAMS)
    inv_q_std = np.std(results['inventory_terminal_q'])
    sym_q_std = np.std(results['symmetric_terminal_q'])
    assert inv_q_std < sym_q_std


def test_run_all_gammas_returns_all_gammas():
    results = run_all_gammas([0.01, 0.1], PARAMS, seed=42)
    assert 0.01 in results
    assert 0.1 in results


def test_compute_stats_keys():
    rng = np.random.default_rng(0)
    results = run_simulation(gamma=0.1, n_paths=20, rng=rng, params=PARAMS)
    stats = compute_stats(results)
    for key in ['inv_mean', 'inv_std', 'sym_mean', 'sym_std', 'inv_q_std', 'sym_q_std']:
        assert key in stats


def test_reproducibility():
    # Same seed -> same results
    r1 = run_all_gammas([0.1], PARAMS, seed=123)
    r2 = run_all_gammas([0.1], PARAMS, seed=123)
    assert np.allclose(r1[0.1]['inventory_pnl'], r2[0.1]['inventory_pnl'])


def test_high_gamma_lower_std_than_low_gamma():
    # Higher gamma -> more aggressive inventory skewing -> lower std
    results_lo = run_all_gammas([0.01], PARAMS, seed=42)
    results_hi = run_all_gammas([0.5], PARAMS, seed=42)
    std_lo = np.std(results_lo[0.01]['inventory_pnl'])
    std_hi = np.std(results_hi[0.5]['inventory_pnl'])
    # Higher gamma should constrain variance more
    assert std_hi < std_lo
