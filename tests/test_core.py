"""Unit tests: run with `pytest -q`. They use fake data with the real schema (tests only)."""
import numpy as np

from src.data import FLAG, HISTORY, TARGET, TRANSACTIONAL, make_synthetic, split_data
from src.metrics import calibration_slope_intercept, ece, fit_calibrator
from src.missingness import (add_noise, corrupt, history_mask, reference_stats,
                             sparsify_transactional)


def _setup(seed=0):
    df = make_synthetic(8000, seed)
    train, val, test = split_data(df, seed)
    return train, val, test, reference_stats(train)


def test_split_is_stratified_and_disjoint_sizes():
    df = make_synthetic(8000, 1)
    tr, va, te = split_data(df, 1)
    assert len(tr) + len(va) + len(te) == len(df)
    rates = [p[TARGET].mean() for p in (tr, va, te)]
    assert max(rates) - min(rates) < 0.01


def test_mcar_and_mar_hit_target_rate():
    train, _, _, stats = _setup()
    rng = np.random.default_rng(0)
    for mech in ("MCAR", "MAR"):
        for rate in (0.2, 0.5, 0.8):
            m = history_mask(train, mech, rate, rng, stats)
            assert abs(m.mean() - rate) < 0.03, (mech, rate, m.mean())


def test_mar_depends_on_credit_limit_but_mcar_does_not():
    train, _, _, stats = _setup()
    rng = np.random.default_rng(1)
    mar = history_mask(train, "MAR", 0.5, rng, stats)
    mcar = history_mask(train, "MCAR", 0.5, rng, stats)
    lim = train["LIMIT_BAL"].to_numpy()
    assert lim[mar].mean() < lim[~mar].mean() * 0.9      # low limit => more often thin-file
    assert abs(lim[mcar].mean() - lim[~mcar].mean()) < 0.05 * lim.mean()


def test_label_and_profile_untouched_and_whole_block_hidden():
    train, _, _, stats = _setup()
    cond = dict(mechanism="MAR", hist_rate=0.5, months=3, noise=0.25)
    out = corrupt(train, cond, np.random.default_rng(2), stats)
    assert (out[TARGET].to_numpy() == train[TARGET].to_numpy()).all()
    assert (out["LIMIT_BAL"].to_numpy() == train["LIMIT_BAL"].to_numpy()).all()
    hidden = out[FLAG].to_numpy() == 1
    assert out.loc[hidden, HISTORY].isna().all().all()      # whole block hidden together
    assert not out.loc[~hidden, HISTORY].isna().any().any()  # nobody else touched


def test_sparsity_keeps_only_recent_months():
    train, _, _, _ = _setup()
    out = sparsify_transactional(train, 2)
    assert out[["BILL_AMT1", "BILL_AMT2", "PAY_AMT1", "PAY_AMT2"]].notna().all().all()
    assert out[["BILL_AMT3", "BILL_AMT6", "PAY_AMT3", "PAY_AMT6"]].isna().all().all()


def test_noise_scale_is_fraction_of_sd_and_mean_preserved():
    train, _, _, stats = _setup()
    out = add_noise(train, 0.5, np.random.default_rng(3), stats["trans_sd"])
    for c in TRANSACTIONAL[:3]:
        eps = out[c] - train[c]
        assert abs(eps.std() / stats["trans_sd"][c] - 0.5) < 0.05
        assert abs(eps.mean()) < 0.05 * stats["trans_sd"][c]


def test_calibration_slope_is_one_for_perfectly_calibrated_predictions():
    rng = np.random.default_rng(4)
    p = rng.uniform(0.02, 0.9, 60000)
    y = (rng.random(60000) < p).astype(int)
    slope, intercept = calibration_slope_intercept(y, p)
    assert abs(slope - 1) < 0.05 and abs(intercept) < 0.05
    assert ece(y, p) < 0.01


def test_recalibration_fixes_a_distorted_model():
    rng = np.random.default_rng(5)
    true_p = rng.uniform(0.02, 0.8, 40000)
    y = (rng.random(40000) < true_p).astype(int)
    distorted = np.clip(true_p ** 0.5, 0.01, 0.99)  # badly overconfident-high predictions
    half = 20000
    for kind in ("platt", "isotonic"):
        f = fit_calibrator(kind, distorted[:half], y[:half])
        assert ece(y[half:], f(distorted[half:])) < ece(y[half:], distorted[half:]) / 2
