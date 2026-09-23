"""
IV Surface Builder — SVI parametrisation for NSE option chains.

Provides:
  fit_svi(strikes, ivs, forward) → dict
      Fit the Stochastic Volatility Inspired (SVI) model to a single
      expiry's IV smile using scipy.optimize.minimize.

  build_iv_surface(snapshots_by_expiry) → dict
      Collect per-expiry IV arrays keyed by expiry label.

  compute_term_structure(atm_ivs_by_expiry) → list[dict]
      Return ATM IV per expiry sorted ascending by days-to-expiry.

SVI model (raw parametrisation)
--------------------------------
  w(k) = a + b * (rho * (k − m) + sqrt((k − m)^2 + sigma^2))

where
  k  = log(K / F)          log-moneyness
  w  = sigma_implied^2     total variance (T=1 convention for fitting)

Bounds used in fitting:
  a     ∈ [1e-6, 1.0]
  b     ∈ [1e-6, 2.0]
  rho   ∈ (−0.999, 0.999)
  m     ∈ [−2.0, 2.0]
  sigma ∈ [1e-6, 2.0]

Validates: Requirements 15.5
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# SVI fitting
# ---------------------------------------------------------------------------


def fit_svi(
    strikes: list[float],
    ivs: list[float],
    forward: float,
) -> dict:
    """
    Fit the SVI raw parametrisation to a single-expiry IV smile.

    The fit minimises the sum-of-squared differences between the SVI
    implied volatility (sqrt of total variance at T=1) and the market
    implied volatilities.

    Parameters
    ----------
    strikes : list of option strike prices (same units as *forward*)
    ivs     : list of market implied volatilities (decimal, e.g. 0.15 for 15%)
    forward : forward price for this expiry (F = S * exp(r * T))

    Returns
    -------
    dict with keys: a, b, rho, m, sigma  (all floats)

    Validates: Requirements 15.5
    """
    strikes_arr = np.asarray(strikes, dtype=float)
    ivs_arr = np.asarray(ivs, dtype=float)

    # Log-moneyness: k = log(K / F)
    k = np.log(strikes_arr / forward)

    # -----------------------------------------------------------------------
    # SVI implied-vol function (T=1 convention — w = IV^2 directly)
    # -----------------------------------------------------------------------
    def _svi_iv(
        k_arr: np.ndarray,
        a: float,
        b: float,
        rho: float,
        m: float,
        sigma: float,
    ) -> np.ndarray:
        w = a + b * (rho * (k_arr - m) + np.sqrt((k_arr - m) ** 2 + sigma ** 2))
        return np.sqrt(np.maximum(w, 1e-10))

    def _objective(params: np.ndarray) -> float:
        a, b, rho, m, sigma = params
        predicted = _svi_iv(k, a, b, rho, m, sigma)
        return float(np.sum((predicted - ivs_arr) ** 2))

    # Initial guess: realistic NIFTY-like starting point
    x0 = np.array([0.04, 0.1, -0.3, 0.0, 0.2])
    bounds = [
        (1e-6, 1.0),       # a
        (1e-6, 2.0),       # b
        (-0.999, 0.999),   # rho
        (-2.0, 2.0),       # m
        (1e-6, 2.0),       # sigma
    ]

    result = minimize(
        _objective,
        x0,
        bounds=bounds,
        method="L-BFGS-B",
        options={"maxiter": 1000, "ftol": 1e-12, "gtol": 1e-8},
    )

    a, b, rho, m, sigma = result.x

    return {
        "a": float(a),
        "b": float(b),
        "rho": float(rho),
        "m": float(m),
        "sigma": float(sigma),
    }


# ---------------------------------------------------------------------------
# IV surface builder
# ---------------------------------------------------------------------------


def build_iv_surface(snapshots_by_expiry: dict) -> dict:
    """
    Collect per-expiry strike/IV arrays into a surface dict.

    Parameters
    ----------
    snapshots_by_expiry : dict keyed by expiry label (e.g. "2025-07-10").
        Each value is a dict with at minimum:
          "strikes" : list[float]
          "ivs"     : list[float]
        Additional keys (forward, days_to_expiry, atm_iv) are passed through
        unchanged but not used by this function.

    Returns
    -------
    dict keyed by expiry label.  Each value is a list of dicts:
      [{"strike": float, "iv": float}, ...]

    Validates: Requirements 15.5
    """
    result: dict = {}
    for expiry, snapshot in snapshots_by_expiry.items():
        strikes = snapshot["strikes"]
        ivs = snapshot["ivs"]
        result[expiry] = [
            {"strike": float(s), "iv": float(iv)}
            for s, iv in zip(strikes, ivs)
        ]
    return result


# ---------------------------------------------------------------------------
# Term structure
# ---------------------------------------------------------------------------


def compute_term_structure(atm_ivs_by_expiry: dict) -> list[dict]:
    """
    Build the ATM IV term structure sorted ascending by days-to-expiry.

    Parameters
    ----------
    atm_ivs_by_expiry : dict keyed by expiry label.
        Each value is a dict with at minimum:
          "days_to_expiry" : int | float
          "atm_iv"         : float

    Returns
    -------
    list of dicts — one entry per expiry, sorted by days_to_expiry ascending:
      [{"expiry": str, "days_to_expiry": int|float, "atm_iv": float}, ...]

    Validates: Requirements 15.5
    """
    entries: list[dict] = []
    for expiry, snapshot in atm_ivs_by_expiry.items():
        entries.append(
            {
                "expiry": expiry,
                "days_to_expiry": snapshot["days_to_expiry"],
                "atm_iv": snapshot["atm_iv"],
            }
        )
    return sorted(entries, key=lambda x: x["days_to_expiry"])
