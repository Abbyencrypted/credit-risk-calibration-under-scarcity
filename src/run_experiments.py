"""Days 6-10: the full experiment grid. One output row per (seed, regime, model, condition, recalibration).

Regime A  "deployment degradation": train on COMPLETE data; corrupt validation/test only.
Regime B  "adaptive": train, validation and test all corrupted by the SAME condition.

Recalibration (Platt / isotonic) is fitted on the (corrupted) VALIDATION set, never on test.
Interpretation: "a small labelled sample from the degraded environment is available to recalibrate".

Usage:
  python -m src.run_experiments                 # full run (5 seeds)
  python -m src.run_experiments --quick         # 2 seeds, fewer conditions (sanity check)
  python -m src.run_experiments --synthetic --quick   # smoke test on fake data (never report!)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import TARGET, load_data, split_data
from .metrics import evaluate, fit_calibrator, reliability_bins
from .missingness import corrupt, reference_stats
from .models import build_pipeline

RESULTS = Path(__file__).resolve().parents[1] / "results"

MODEL_CONFIGS = [("l2", 1.0), ("l1", 0.1)]
RECALS = ["none", "platt", "isotonic"]


def build_conditions(quick: bool = False):
    """Condition = history-block missingness x transactional sparsity x transactional noise."""
    conds, seen = [], set()

    def add(mech, rate, months, noise):
        key = (mech, rate, months, noise)
        if key not in seen:
            seen.add(key)
            conds.append(dict(mechanism=mech, hist_rate=rate, months=months, noise=noise))

    # Grid 1: how much history is missing, and how (MCAR vs MAR)
    add("none", 0.0, 6, 0.0)
    for mech in ["MCAR", "MAR"]:
        for r in ([0.5] if quick else [0.2, 0.5, 0.8]):
            add(mech, r, 6, 0.0)
    add("ALL", 1.0, 6, 0.0)
    # Grid 2: transactional scarcity (sparsity x noise) under three history scenarios
    scenarios = [("none", 0.0), ("MAR", 0.5), ("ALL", 1.0)]
    months_grid = [6, 1] if quick else [6, 3, 1]
    noise_grid = [0.0, 0.5] if quick else [0.0, 0.25, 0.5]
    for mech, r in scenarios:
        for m in months_grid:
            for nz in noise_grid:
                add(mech, r, m, nz)
    return conds


def variants(p_val, y_val, p_te):
    yield "none", p_te
    for kind in ("platt", "isotonic"):
        yield kind, fit_calibrator(kind, p_val, y_val)(p_te)


def run(seeds: int = 5, quick: bool = False, data: str | None = None, synthetic: bool = False,
        out: str | None = None, df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Run the full grid and write results.csv (+ calibration_bins.csv). Callable from a notebook."""
    out = Path(out) if out else RESULTS / "results.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_data(data, synthetic)
    conds = build_conditions(quick)
    print(f"{len(df):,} rows | {len(conds)} conditions | {seeds} seeds | quick={quick}", flush=True)

    rows, bins_rows = [], []
    t0 = time.time()
    for seed in range(seeds):
        train, val, test = split_data(df, seed)
        y_val, y_te = val[TARGET].to_numpy(), test[TARGET].to_numpy()
        stats = reference_stats(train)

        for reg, C in MODEL_CONFIGS:
            # Reference model for the stability measures: trained and tested on COMPLETE data.
            ref = build_pipeline(reg, C, use_flag=False).fit(train, train[TARGET])
            p_ref = ref.predict_proba(test)[:, 1]

            for ci, cond in enumerate(conds):
                def rng_for(part, seed=seed, ci=ci):  # reproducible, independent stream per (seed, cond, part)
                    return np.random.default_rng([seed, ci, part])

                val_c = corrupt(val, cond, rng_for(1), stats)
                te_c = corrupt(test, cond, rng_for(2), stats)
                realised = float(te_c["hist_missing"].mean())

                # ---- Regime A: reuse the clean-trained model ----
                p_val = ref.predict_proba(val_c)[:, 1]
                p_te = ref.predict_proba(te_c)[:, 1]
                _record(rows, bins_rows, seed, "A", reg, C, False, cond, realised,
                        y_val, y_te, p_val, p_te, p_ref)

                # ---- Regime B: train on the same degraded condition ----
                tr_c = corrupt(train, cond, rng_for(0), stats)
                for use_flag in (False, True):
                    m = build_pipeline(reg, C, use_flag=use_flag).fit(tr_c, train[TARGET])
                    p_val = m.predict_proba(val_c)[:, 1]
                    p_te = m.predict_proba(te_c)[:, 1]
                    _record(rows, bins_rows, seed, "B", reg, C, use_flag, cond, realised,
                            y_val, y_te, p_val, p_te, p_ref)
        print(f"seed {seed + 1}/{seeds} done  ({time.time() - t0:.0f}s elapsed)", flush=True)
        pd.DataFrame(rows).to_csv(out, index=False)  # incremental save: safe if you stop early

    res = pd.DataFrame(rows)
    res.to_csv(out, index=False)
    pd.DataFrame(bins_rows).to_csv(out.with_name("calibration_bins.csv"), index=False)
    print(f"Wrote {len(res):,} rows to {out}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--data", default=None)
    ap.add_argument("--synthetic", action="store_true", help="smoke test only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    seeds = 2 if a.quick and a.seeds == 5 else a.seeds
    run(seeds, a.quick, a.data, a.synthetic, a.out)


def _record(rows, bins_rows, seed, regime, reg, C, use_flag, cond, realised, y_val, y_te, p_val, p_te, p_ref):
    for recal, p in variants(p_val, y_val, p_te):
        base = dict(seed=seed, regime=regime, reg=reg, C=C, use_flag=use_flag, recal=recal,
                    mechanism=cond["mechanism"], hist_rate=cond["hist_rate"], months=cond["months"],
                    noise=cond["noise"], realised_hist_missing=realised)
        rows.append({**base, **evaluate(y_te, p, p_ref)})
        if seed == 0:  # keep reliability-diagram data for one seed only (keeps the file small)
            b = reliability_bins(y_te, p)
            for r in b.to_dict("records"):
                bins_rows.append({**base, **r})


if __name__ == "__main__":
    main()
