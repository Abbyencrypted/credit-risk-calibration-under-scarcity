"""Days 2-3: full-information baseline, and 'how much does history matter?'

Usage:  python -m src.run_baseline [--seeds 5] [--synthetic]
Writes: results/baseline.csv  (+ prints a mean +/- SD table)
"""
import argparse
from pathlib import Path

import pandas as pd

from .data import TARGET, load_data, split_data
from .metrics import evaluate
from .models import build_pipeline

RESULTS = Path(__file__).resolve().parents[1] / "results"

FEATURE_SETS = {
    "full (history + transactional + profile)": dict(),
    "no history (transactional + profile)": dict(drop_history=True),
    "profile only": dict(drop_history=True, drop_transactional=True),
}


def run(seeds: int = 5, data: str | None = None, synthetic: bool = False, df=None):
    """Returns (per-seed results, mean/SD summary). Callable from a notebook."""
    if df is None:
        df = load_data(data, synthetic)
    rows = []
    for seed in range(seeds):
        train, val, test = split_data(df, seed)
        for name, kw in FEATURE_SETS.items():
            model = build_pipeline("l2", 1.0, **kw).fit(train, train[TARGET])
            p = model.predict_proba(test)[:, 1]
            rows.append({"seed": seed, "features": name, **evaluate(test[TARGET], p)})
    res = pd.DataFrame(rows)
    RESULTS.mkdir(exist_ok=True)
    res.to_csv(RESULTS / "baseline.csv", index=False)
    cols = ["auc", "pr_auc", "log_loss", "brier", "cal_slope", "ece"]
    summary = res.groupby("features", sort=False)[cols].agg(["mean", "std"]).round(4)
    return res, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--data", default=None)
    ap.add_argument("--synthetic", action="store_true", help="smoke test only")
    a = ap.parse_args()
    df = load_data(a.data, a.synthetic)
    print(f"Loaded {len(df):,} rows, default rate {df[TARGET].mean():.3f}")
    _, summary = run(a.seeds, df=df)
    print(summary.to_string())


if __name__ == "__main__":
    main()
