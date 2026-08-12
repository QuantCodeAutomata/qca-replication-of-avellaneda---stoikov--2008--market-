import numpy as np

# Custom — Context7 found no library equivalent (AS 2008 closed-form solutions)

PARAMS: dict = {
    "S0": 100.0,       # initial mid-price
    "SIGMA": 2.0,      # volatility
    "T": 1.0,          # total time horizon
    "DT": 0.005,       # time step (200 steps)
    "K": 1.5,          # order arrival shape
    "A": 140.0,        # order arrival scale
    "Q0": 0,           # initial inventory
    "N_PATHS": 1000,   # number of Monte Carlo paths
    "GAMMAS": [0.01, 0.1, 0.5],  # risk-aversion levels
}

GAMMAS: list = PARAMS["GAMMAS"]


def indifference_price(
    s: float, q: int, gamma: float, sigma: float, T: float, t: float
) -> float:
    """Reservation (indifference) price per AS 2008 Eq. (12).

    Parameters
    ----------
    s : float
        Current mid-price.
    q : int
        Current inventory level.
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Volatility of the mid-price.
    T : float
        Total time horizon.
    t : float
        Current time.

    Returns
    -------
    float
        Reservation price adjusted for inventory risk.
    """
    return s - q * gamma * sigma**2 * (T - t)


def optimal_spread(
    gamma: float, sigma: float, T: float, t: float, k: float
) -> float:
    """Optimal bid-ask spread per AS 2008 asymptotic closed-form solution.

    Parameters
    ----------
    gamma : float
        Risk-aversion coefficient.
    sigma : float
        Volatility of the mid-price.
    T : float
        Total time horizon.
    t : float
        Current time. Must be strictly less than T to avoid degenerate spreads;
        caller must ensure this.
    k : float
        Order arrival shape parameter.

    Returns
    -------
    float
        Optimal bid-ask spread.
    """
    return gamma * sigma**2 * (T - t) + (2.0 / gamma) * np.log(1.0 + gamma / k)


def arrival_intensity(delta: float, A: float, k: float) -> float:
    """Poisson arrival intensity for a side at queue distance delta.

    Parameters
    ----------
    delta : float
        Distance from mid-price to the quote (always non-negative).
    A : float
        Order arrival scale parameter.
    k : float
        Order arrival shape parameter.

    Returns
    -------
    float
        Poisson arrival intensity.
    """
    return A * np.exp(-k * delta)


def execution_prob(lam: float, dt: float) -> float:
    """Execution probability for a given arrival intensity over interval dt.

    Parameters
    ----------
    lam : float
        Poisson arrival intensity.
    dt : float
        Time step duration.

    Returns
    -------
    float
        Execution probability, capped at 1.0.
    """
    return min(lam * dt, 1.0)
