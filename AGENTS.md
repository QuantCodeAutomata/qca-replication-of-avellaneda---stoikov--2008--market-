# Avellaneda-Stoikov (2008) Replication — Agent Notes

## Project Structure
```
data/README.md         # Notes on synthetic data (no external downloads needed)
src/model.py           # AS 2008 closed-form formulas + PARAMS dict
src/simulation.py      # Monte Carlo engine (run_simulation, run_all_gammas, compute_stats)
src/__init__.py
exp/run_experiment.py  # Full experiment runner: runs 1000 paths, saves CSV + 5 figures to results/
exp/__init__.py
tests/test_model.py    # 11 unit tests for model formulas
tests/test_simulation.py  # 8 integration/statistical tests for simulation
tests/__init__.py
requirements.txt       # numpy>=1.26, matplotlib>=3.7, pandas>=2.0, pytest>=7.0
.gitignore
README.md
```

## Key APIs

### src/model.py
- `PARAMS` dict — keys: S0, SIGMA, T, DT, K, A, Q0, N_PATHS, GAMMAS
- `indifference_price(s, q, gamma, sigma, T, t)` → float  (r = s - q·γ·σ²·(T-t))
- `optimal_spread(gamma, sigma, T, t, k)` → float  (γσ²(T-t) + (2/γ)ln(1+γ/k))
- `arrival_intensity(delta, A, k)` → float  (A·exp(-k·delta))
- `execution_prob(lam, dt)` → float  (min(lam·dt, 1.0))

### src/simulation.py
- `run_simulation(gamma, n_paths, rng, params)` → dict with keys:
  inventory_pnl, symmetric_pnl, inventory_terminal_q, symmetric_terminal_q
- `run_all_gammas(gammas, params, seed=42)` → dict mapping gamma → results dict
- `compute_stats(results)` → dict with keys:
  inv_mean, inv_std, sym_mean, sym_std, inv_q_mean, inv_q_std, sym_q_mean, sym_q_std

## Running Tests
```bash
pytest tests/ -v   # 19 tests, all should pass in ~24 s
```

## Design Notes
- Price paths are Arithmetic Brownian Motion (binomial ±σ√dt per step, not geometric)
- Inventory and Symmetric strategies share the same mid-price path per run for fair comparison
- `run_all_gammas` uses the same `seed` for every gamma (independent but reproducible)
- `execution_prob` uses linear approximation: P = lam·dt (capped at 1.0), not 1-exp(-lam·dt)
- Terminal P&L = cash + q·s_final (marked to market)
