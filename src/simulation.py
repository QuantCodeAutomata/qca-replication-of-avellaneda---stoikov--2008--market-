import numpy as np
from typing import Dict, List

from src.model import (
    arrival_intensity,
    execution_prob,
    indifference_price,
    optimal_spread,
    PARAMS,
)


def run_simulation(
    gamma: float,
    n_paths: int,
    rng: np.random.Generator,
    params: dict,
) -> Dict[str, np.ndarray]:
    """Run N Monte Carlo paths for both Inventory and Symmetric strategies
    simultaneously, sharing the same mid-price paths for fair comparison.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    n_paths : int
        Number of Monte Carlo paths to simulate.
    rng : np.random.Generator
        Seeded random number generator.
    params : dict
        Parameter dictionary with keys: S0, SIGMA, T, DT, K, A, Q0.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary with keys 'inventory_pnl', 'symmetric_pnl',
        'inventory_terminal_q', 'symmetric_terminal_q'.
    """
    S0 = params["S0"]
    SIGMA = params["SIGMA"]
    T = params["T"]
    DT = params["DT"]
    K = params["K"]
    A = params["A"]
    Q0 = params["Q0"]

    n_steps = round(T / DT)  # = 200

    inventory_pnl = np.empty(n_paths)
    symmetric_pnl = np.empty(n_paths)
    inventory_terminal_q = np.empty(n_paths)
    symmetric_terminal_q = np.empty(n_paths)

    for p in range(n_paths):
        # Shared mid-price path
        s = S0
        # Separate state for each strategy
        x_inv, q_inv = 0.0, Q0     # inventory strategy: cash, inventory
        x_sym, q_sym = 0.0, Q0     # symmetric strategy: cash, inventory

        for i in range(n_steps):
            t = i * DT  # current time; strictly < T because i < n_steps

            # --- Compute optimal spread (shared formula, different quotes) ---
            spread = optimal_spread(gamma, SIGMA, T, t, K)

            # --- Inventory strategy quotes ---
            r = indifference_price(s, q_inv, gamma, SIGMA, T, t)
            pa_inv = r + spread / 2.0
            pb_inv = r - spread / 2.0

            # --- Symmetric strategy quotes ---
            pa_sym = s + spread / 2.0
            pb_sym = s - spread / 2.0

            # --- Arrival intensities (distance = quote - mid for ask; mid - quote for bid) ---
            lam_a_inv = arrival_intensity(pa_inv - s, A, K)
            lam_b_inv = arrival_intensity(s - pb_inv, A, K)
            lam_a_sym = arrival_intensity(pa_sym - s, A, K)
            lam_b_sym = arrival_intensity(s - pb_sym, A, K)

            # --- Execution probabilities ---
            prob_a_inv = execution_prob(lam_a_inv, DT)
            prob_b_inv = execution_prob(lam_b_inv, DT)
            prob_a_sym = execution_prob(lam_a_sym, DT)
            prob_b_sym = execution_prob(lam_b_sym, DT)

            # --- Simulate fills using uniform random draws ---
            # Draw 4 independent uniforms per step per path
            u = rng.uniform(size=4)

            # Inventory strategy fills (ask first, then bid)
            if u[0] < prob_a_inv:  # ask filled: sell 1 unit
                x_inv += pa_inv
                q_inv -= 1
            if u[1] < prob_b_inv:  # bid filled: buy 1 unit
                x_inv -= pb_inv
                q_inv += 1

            # Symmetric strategy fills (ask first, then bid)
            if u[2] < prob_a_sym:  # ask filled: sell 1 unit
                x_sym += pa_sym
                q_sym -= 1
            if u[3] < prob_b_sym:  # bid filled: buy 1 unit
                x_sym -= pb_sym
                q_sym += 1

            # --- Update mid-price: binomial random walk ---
            sign = 1 if rng.random() < 0.5 else -1
            s += sign * SIGMA * np.sqrt(DT)

        # --- Terminal P&L = cash + inventory * final mid-price ---
        inventory_pnl[p] = x_inv + q_inv * s
        symmetric_pnl[p] = x_sym + q_sym * s
        inventory_terminal_q[p] = q_inv
        symmetric_terminal_q[p] = q_sym

    return {
        "inventory_pnl": inventory_pnl,
        "symmetric_pnl": symmetric_pnl,
        "inventory_terminal_q": inventory_terminal_q,
        "symmetric_terminal_q": symmetric_terminal_q,
    }


def run_all_gammas(
    gammas: List[float],
    params: dict,
    seed: int = 42,
) -> Dict[float, Dict[str, np.ndarray]]:
    """Run simulation for each gamma value in gammas.

    Uses a fresh seeded RNG per gamma so results are reproducible but
    independent across gammas.

    Parameters
    ----------
    gammas : List[float]
        List of risk-aversion coefficients to simulate.
    params : dict
        Parameter dictionary passed to each simulation run.
    seed : int, optional
        Base seed for reproducibility (default 42).

    Returns
    -------
    Dict[float, Dict[str, np.ndarray]]
        Dictionary mapping gamma -> results dict.
    """
    n_paths = params["N_PATHS"]
    results = {}
    for gamma in gammas:
        rng = np.random.default_rng(seed)
        results[gamma] = run_simulation(gamma, n_paths, rng, params)
    return results


def compute_stats(results: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Compute summary statistics from simulation results.

    Parameters
    ----------
    results : Dict[str, np.ndarray]
        Results dict as returned by run_simulation, containing keys:
        'inventory_pnl', 'symmetric_pnl', 'inventory_terminal_q',
        'symmetric_terminal_q'.

    Returns
    -------
    Dict[str, float]
        Summary statistics with keys:
        inv_mean, inv_std, sym_mean, sym_std,
        inv_q_mean, inv_q_std, sym_q_mean, sym_q_std.
    """
    return {
        "inv_mean": float(np.mean(results["inventory_pnl"])),
        "inv_std": float(np.std(results["inventory_pnl"])),
        "sym_mean": float(np.mean(results["symmetric_pnl"])),
        "sym_std": float(np.std(results["symmetric_pnl"])),
        "inv_q_mean": float(np.mean(results["inventory_terminal_q"])),
        "inv_q_std": float(np.std(results["inventory_terminal_q"])),
        "sym_q_mean": float(np.mean(results["symmetric_terminal_q"])),
        "sym_q_std": float(np.std(results["symmetric_terminal_q"])),
    }
