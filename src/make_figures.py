"""Days 8, 11: turn results.csv into figures and a summary table.

Usage:  python -m src.make_figures [--results results/results.csv] [--outdir results/figures]
Headline settings (everything else is a robustness check): L2 (C=1), no missingness flag, no recalibration.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = Path(__file__).resolve().parents[1] / "results"
METRICS = {"auc": "ROC-AUC", "brier": "Brier score", "cal_slope": "Calibration slope (ideal = 1)",
           "ece": "Expected calibration error", "mpd": "Mean probability displacement",
           "spearman_vs_full": "Rank correlation with full-info model"}
COL = {"MCAR": "#1f77b4", "MAR": "#d62728"}


def headline(df, regime, reg="l2", flag=False, recal="none"):
    return df[(df.regime == regime) & (df.reg == reg) & (df.use_flag == flag) & (df.recal == recal)]


def agg(d, by, metric):
    g = d.groupby(by)[metric].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g


def hist_curve(d, mech):
    """History-missingness curve for one mechanism, months=6, noise=0; endpoints shared by both."""
    d = d[(d.months == 6) & (d.noise == 0)]
    return d[(d.mechanism == mech) | (d.mechanism == "none") | (d.mechanism == "ALL")]


def fig_history(df, regime, outdir):
    d = headline(df, regime)
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    for ax, m in zip(axes, ["auc", "brier", "cal_slope", "ece"]):
        for mech in ("MCAR", "MAR"):
            g = agg(hist_curve(d, mech), "hist_rate", m)
            ax.plot(g.hist_rate, g["mean"], "-o", color=COL[mech], label=mech)
            ax.fill_between(g.hist_rate, g["mean"] - g["std"], g["mean"] + g["std"], color=COL[mech], alpha=0.15)
        if m == "cal_slope":
            ax.axhline(1, color="grey", ls=":")
        ax.set_xlabel("Share of borrowers with no credit history")
        ax.set_title(METRICS[m], fontsize=10)
    axes[0].legend()
    label = "deployment degradation (train complete)" if regime == "A" else "adaptive (train degraded)"
    fig.suptitle(f"Regime {regime}: {label}. Mean +/- SD over seeds", y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / f"fig1_history_missingness_regime{regime}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_calibration_curves(bins, outdir):
    b = bins[(bins.regime == "A") & (bins.reg == "l2") & (~bins.use_flag) & (bins.recal == "none")
             & (bins.months == 6) & (bins.noise == 0)]
    scen = [("none", 0.0, "Complete information"), ("MAR", 0.5, "50% thin-file (MAR)"),
            ("ALL", 1.0, "No history for anyone")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
    for ax, (mech, r, title) in zip(axes, scen):
        s = b[(b.mechanism == mech) & (b.hist_rate == r)].sort_values("bin")
        ax.plot([0, 0.8], [0, 0.8], "k:", label="perfect")
        ax.plot(s.mean_pred, s.frac_pos, "-o", color="#d62728", label="model")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Predicted default probability")
    axes[0].set_ylabel("Observed default rate")
    axes[0].legend()
    fig.suptitle("Reliability diagrams, Regime A, seed 0 (equal-count bins)", y=1.02)
    fig.tight_layout()
    fig.savefig(outdir / "fig2_calibration_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_sparsity_noise(df, outdir):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for row, (regime, rname) in enumerate([("A", "Regime A"), ("B", "Regime B")]):
        d = headline(df, regime)
        d = d[(d.mechanism == "ALL") & (d.hist_rate == 1.0)]
        for ax, m in zip(axes[row], ["auc", "brier", "cal_slope"]):
            pv = d.groupby(["months", "noise"])[m].mean().unstack("noise").sort_index(ascending=False)
            im = ax.imshow(pv.values, cmap="viridis", aspect="auto")
            ax.set_xticks(range(len(pv.columns)), [f"{c:g}" for c in pv.columns])
            ax.set_yticks(range(len(pv.index)), [f"{i}" for i in pv.index])
            ax.set_xlabel("Noise (fraction of SD)")
            ax.set_ylabel("Months of transactional data kept")
            ax.set_title(f"{rname}: {METRICS[m]}", fontsize=10)
            for i in range(pv.shape[0]):
                for j in range(pv.shape[1]):
                    ax.text(j, i, f"{pv.values[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("No credit history for anyone: effect of sparse and noisy transactional data", y=1.0)
    fig.tight_layout()
    fig.savefig(outdir / "fig3_sparsity_noise.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_recalibration(df, outdir):
    d = df[(df.regime == "A") & (df.reg == "l2") & (~df.use_flag) & (df.months == 6) & (df.noise == 0)]
    d = d[(d.mechanism == "ALL") & (d.hist_rate == 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6))
    for ax, m in zip(axes, ["auc", "brier", "cal_slope", "ece"]):
        g = agg(d, "recal", m).set_index("recal").loc[["none", "platt", "isotonic"]]
        ax.bar(g.index, g["mean"], yerr=g["std"], color=["#999", "#2ca02c", "#9467bd"], capsize=3)
        if m == "cal_slope":
            ax.axhline(1, color="k", ls=":")
        ax.set_title(METRICS[m], fontsize=10)
    fig.suptitle("Recalibration when nobody has credit history (Regime A, fitted on degraded validation data)", y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "fig4_recalibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_stability(df, outdir):
    d = headline(df, "A")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for ax, m in zip(axes, ["mpd", "spearman_vs_full"]):
        for mech in ("MCAR", "MAR"):
            g = agg(hist_curve(d, mech), "hist_rate", m)
            ax.plot(g.hist_rate, g["mean"], "-o", color=COL[mech], label=mech)
            ax.fill_between(g.hist_rate, g["mean"] - g["std"], g["mean"] + g["std"], color=COL[mech], alpha=0.15)
        ax.set_xlabel("Share of borrowers with no credit history")
        ax.set_title(METRICS[m], fontsize=10)
    axes[0].legend()
    fig.suptitle("Prediction stability vs the complete-information model (Regime A)", y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "fig5_stability.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_design_choices(df, outdir):
    """Regime B, MAR: L1 vs L2 and with/without the 'history missing' flag."""
    d = df[(df.regime == "B") & (df.recal == "none") & (df.months == 6) & (df.noise == 0)]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for ax, m in zip(axes, ["auc", "brier", "ece"]):
        for (reg, flag), ls in [(("l2", False), "-"), (("l2", True), "--"), (("l1", False), "-."), (("l1", True), ":")]:
            s = d[(d.reg == reg) & (d.use_flag == flag)]
            g = agg(hist_curve(s, "MAR"), "hist_rate", m)
            ax.plot(g.hist_rate, g["mean"], ls, marker="o", ms=3, label=f"{reg.upper()}, flag={'on' if flag else 'off'}")
        ax.set_xlabel("Share of borrowers with no credit history (MAR)")
        ax.set_title(METRICS[m], fontsize=10)
    axes[0].legend(fontsize=8)
    fig.suptitle("Design choices in Regime B: penalty type and missingness indicator", y=1.03)
    fig.tight_layout()
    fig.savefig(outdir / "fig6_design_choices.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def summary_table(df, outpath):
    cols = ["auc", "brier", "cal_slope", "ece", "mpd"]
    d = headline(df, "A")
    d = d[(d.months == 6) & (d.noise == 0)].copy()
    d["scenario"] = d.mechanism + " " + d.hist_rate.astype(str)
    t = d.groupby("scenario")[cols].agg(["mean", "std"]).round(4)
    t.to_csv(outpath)
    return t


def make_all(results_path=None, outdir=None, show_table=True):
    """Make every figure and the summary table. Callable from a notebook."""
    results_path = Path(results_path) if results_path else RES / "results.csv"
    outdir = Path(outdir) if outdir else RES / "figures"
    df = pd.read_csv(results_path)
    outdir.mkdir(parents=True, exist_ok=True)
    bins_path = results_path.with_name("calibration_bins.csv")
    for regime in ("A", "B"):
        fig_history(df, regime, outdir)
    if bins_path.exists():
        fig_calibration_curves(pd.read_csv(bins_path), outdir)
    fig_sparsity_noise(df, outdir)
    fig_recalibration(df, outdir)
    fig_stability(df, outdir)
    fig_design_choices(df, outdir)
    table = summary_table(df, results_path.with_name("summary_table.csv"))
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RES / "results.csv"))
    ap.add_argument("--outdir", default=str(RES / "figures"))
    a = ap.parse_args()
    table = make_all(a.results, a.outdir)
    print(table.to_string())
    print(f"Figures written to {a.outdir}")


if __name__ == "__main__":
    main()
