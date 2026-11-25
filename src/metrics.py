"""Discrimination, calibration and stability metrics, plus post-hoc recalibration."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

EPS = 1e-6


def _logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y, p):
    """Fit logit P(Y=1) = a + b * logit(p_hat). Ideal calibration: b = 1, a = 0.
    b < 1 means predictions are too extreme (overconfident); b > 1 too timid."""
    m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(p).reshape(-1, 1), y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def reliability_bins(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-count bins of predicted probability: mean prediction vs observed default rate."""
    y, p = np.asarray(y), np.asarray(p)
    order = np.argsort(p, kind="stable")
    rows = []
    for k, idx in enumerate(np.array_split(order, bins)):
        rows.append({"bin": k, "mean_pred": p[idx].mean(), "frac_pos": y[idx].mean(), "n": len(idx)})
    return pd.DataFrame(rows)


def ece(y, p, bins: int = 10) -> float:
    """Expected calibration error with equal-count bins: sum_k (n_k/n) * |observed_k - predicted_k|."""
    b = reliability_bins(y, p, bins)
    return float(((b["n"] / b["n"].sum()) * (b["frac_pos"] - b["mean_pred"]).abs()).sum())


def evaluate(y, p, p_ref=None) -> dict:
    """Metrics for predicted probabilities p. p_ref = predictions of the complete-information model
    on the same test borrowers; if given, stability measures are added."""
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    slope, intercept = calibration_slope_intercept(y, p)
    out = {
        "auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "log_loss": log_loss(y, np.clip(p, EPS, 1 - EPS)),
        "brier": brier_score_loss(y, p),
        "cal_slope": slope,
        "cal_intercept": intercept,
        "ece": ece(y, p),
        "mean_pred": float(p.mean()),
    }
    if p_ref is not None:
        p_ref = np.asarray(p_ref, dtype=float)
        out["mpd"] = float(np.mean(np.abs(p - p_ref)))  # mean probability displacement (project-specific)
        rho = spearmanr(p, p_ref).statistic if np.ptp(p) > 0 else np.nan
        out["spearman_vs_full"] = float(rho)
    return out


def fit_calibrator(kind: str, p_val, y_val):
    """Return a function mapping raw probabilities to recalibrated ones, fitted on VALIDATION data only."""
    if kind == "platt":
        m = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(np.asarray(p_val)).reshape(-1, 1), y_val)
        return lambda p: m.predict_proba(_logit(np.asarray(p)).reshape(-1, 1))[:, 1]
    if kind == "isotonic":
        m = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(p_val, y_val)
        return lambda p: np.clip(m.predict(p), EPS, 1 - EPS)
    raise ValueError(kind)
