"""Simulated information loss: block missingness (MCAR / MAR), transactional sparsity, noise.

Notation (see README): X = (H, T, P) with history H, transactional T, profile P.
A borrower's *history block* is hidden as a whole ("thin file"): X_obs = M * X with M_i = 0 on all of H.

Important: the label is never touched, and MAR depends only on an OBSERVED profile variable
(credit limit), so it is a genuine MAR mechanism. We do NOT simulate MNAR in the core study.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import expit

from .data import FLAG, HISTORY, TRANSACTIONAL


def reference_stats(train: pd.DataFrame) -> dict:
    """Statistics from the CLEAN training set, used to scale MAR drivers and noise (no test leakage)."""
    return {
        "limit_mean": float(train["LIMIT_BAL"].mean()),
        "limit_std": float(train["LIMIT_BAL"].std()),
        "trans_sd": train[TRANSACTIONAL].std().to_dict(),
    }


def history_mask(df: pd.DataFrame, mechanism: str, rate: float, rng: np.random.Generator,
                 stats: dict, gamma: float = 1.5) -> np.ndarray:
    """Boolean array, True = this borrower's whole history block is hidden.

    mechanism:
      'none' / rate 0 -> nobody hidden
      'ALL'  / rate 1 -> everybody hidden
      'MCAR' -> P(hidden) = rate for everyone, independent of all data
      'MAR'  -> P(hidden_i) = sigmoid(a + gamma * z_i), z_i = standardised NEGATIVE credit limit
                (lower limit => more likely thin-file). `a` is solved so the average rate = `rate`.
    """
    n = len(df)
    if mechanism == "none" or rate <= 0:
        return np.zeros(n, dtype=bool)
    if mechanism == "ALL" or rate >= 1:
        return np.ones(n, dtype=bool)
    if mechanism == "MCAR":
        return rng.random(n) < rate
    if mechanism == "MAR":
        z = -(df["LIMIT_BAL"].to_numpy() - stats["limit_mean"]) / stats["limit_std"]
        a = brentq(lambda a_: expit(a_ + gamma * z).mean() - rate, -50, 50)
        return rng.random(n) < expit(a + gamma * z)
    raise ValueError(f"Unknown mechanism: {mechanism}")


def apply_history_mask(df: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out.loc[mask, HISTORY] = np.nan
    out[FLAG] = mask.astype(int)
    return out


def sparsify_transactional(df: pd.DataFrame, months_kept: int) -> pd.DataFrame:
    """Keep only the most recent `months_kept` months of bill/payment amounts (month 1 = most recent)."""
    out = df.copy()
    for m in range(months_kept + 1, 7):
        out[f"BILL_AMT{m}"] = np.nan
        out[f"PAY_AMT{m}"] = np.nan
    return out


def add_noise(df: pd.DataFrame, level: float, rng: np.random.Generator, trans_sd: dict) -> pd.DataFrame:
    """Gaussian measurement error on transactional amounts: x* = x + eps, eps ~ N(0, (level*SD_j)^2).

    `level` is a fraction of the (clean training) standard deviation of each column, so it is unit-free.
    """
    out = df.copy()
    if level <= 0:
        return out
    for c in TRANSACTIONAL:
        out[c] = out[c] + rng.normal(0.0, level * trans_sd[c], len(out))  # NaN + noise stays NaN
    return out


def corrupt(df: pd.DataFrame, cond: dict, rng: np.random.Generator, stats: dict) -> pd.DataFrame:
    """Apply a full condition: history block missingness, transactional sparsity, then noise."""
    mask = history_mask(df, cond["mechanism"], cond["hist_rate"], rng, stats)
    out = apply_history_mask(df, mask)
    out = sparsify_transactional(out, cond["months"])
    out = add_noise(out, cond["noise"], rng, stats["trans_sd"])
    return out
