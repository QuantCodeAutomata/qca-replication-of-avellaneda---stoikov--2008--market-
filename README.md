# Replication of Avellaneda & Stoikov (2008) Market-Making Model

## Overview
This repository replicates the asymptotic closed-form solutions for optimal bid/ask quotes
from the seminal market-making paper by Avellaneda & Stoikov (2008).

## Research Objectives
1. Demonstrate that a market maker incorporating inventory risk achieves lower P&L variance.
2. Validate the asymptotic closed-form solutions via Monte Carlo simulation (1,000 paths).
3. Evaluate sensitivity to risk-aversion parameter gamma in {0.01, 0.1, 0.5}.

## Methodology
- Arithmetic Brownian Motion for mid-price (s0=100, sigma=2, T=1, dt=0.005)
- Indifference price: r = s - q * gamma * sigma^2 * (T - t)
- Optimal spread: gamma*sigma^2*(T-t) + (2/gamma)*ln(1 + gamma/k)
- Poisson order arrivals with intensity A*exp(-k*delta)
- 1,000 independent paths; Inventory vs Symmetric strategies share same price paths

## Repository Structure
```
data/          # Synthetic data notes
src/           # Core model and simulation
  model.py     # AS 2008 formulas (indifference price, spread, intensities)
  simulation.py # Monte Carlo simulator
exp/           # Experiment runner
  run_experiment.py
tests/         # Pytest suite
results/       # Figures, metrics table, RESULTS.md
```

## Usage
```bash
# Install dependencies
pip install numpy matplotlib pytest

# Run experiment
python exp/run_experiment.py

# Run tests
pytest tests/ -v
```

## Expected Outcomes
| gamma | Inv Std (P&L) | Sym Std (P&L) |
|-------|--------------|---------------|
| 0.01  | ~similar     | ~similar      |
| 0.10  | ~5.89        | ~13.43        |
| 0.50  | ~4.72        | much higher   |

## Reference
Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.*
Quantitative Finance, 8(3), 217–224.
