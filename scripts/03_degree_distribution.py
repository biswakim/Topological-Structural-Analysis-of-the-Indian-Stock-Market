#!/usr/bin/env python
"""Analysis A - degree distribution: linear and log-log plots, and a
statistical test of the "scale-free" hypothesis."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from nsenet import fits as F  # noqa: E402
from nsenet import plotting as P  # noqa: E402
from nsenet.config import load_config  # noqa: E402
from nsenet.metrics import degree_distribution, degree_series, log_binned_pdf  # noqa: E402
from nsenet.pipeline import prepare, save_numbers, save_table  # noqa: E402

plt = P.plt
MODEL_STYLE = {
    "power law": (P.RED, "-"),
    "exponential": (P.AQUA, "--"),
    "log-normal": (P.VIOLET, "-."),
    "power law + cut-off": (P.ORANGE, ":"),
}


def analyse(g: nx.Graph, seed: int, n_boot: int) -> dict:
    """Everything we want to know about one network's degree sequence."""
    k = degree_series(g)                      # nx.degree() under the hood
    n = len(k)
    pos = k[k > 0].to_numpy()
    gof = F.powerlaw_gof(pos, n_boot=n_boot, seed=seed)
    kmin = gof["kmin"]
    tail_fits = F.fit_tail(pos, kmin)
    table = F.comparison_table(tail_fits)
    full_fits = F.fit_tail(pos, 1)            # same models on the whole k >= 1 range
    full_table = F.comparison_table(full_fits)
    p_er = k.mean() / (n - 1)
    summary = {
        "n": n, "edges": int(k.sum() // 2), "mean": k.mean(), "median": float(k.median()), "max": int(k.max()),
        "sd": k.std(ddof=1), "isolated": int((k == 0).sum()), "cv": k.std(ddof=1) / k.mean(),
        "kappa": float((k**2).mean() / k.mean()), "skew": float(stats.skew(k)),
        "er_sd": float(np.sqrt((n - 1) * p_er * (1 - p_er))),
        "er_p_isolated": float((1 - p_er) ** (n - 1)),
        "er_max_expected": float(stats.binom.ppf(1 - 1 / n, n - 1, p_er)),
        "top1pct_share": float(np.sort(k)[::-1][: max(n // 100, 1)].sum() / k.sum()),
        "top10pct_share": float(np.sort(k)[::-1][: n // 10].sum() / k.sum()),
        "gini": float(gini(k.to_numpy())),
    }
    return {"k": k, "gof": gof, "tail_fits": tail_fits, "tail_table": table, "full_fits": full_fits,
            "full_table": full_table, "summary": summary, "p_er": p_er}


def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0


def draw_row(axes, res: dict, r: float) -> None:
    k = res["k"]
    n = len(k)
    dist = degree_distribution(k)
    kmin = res["gof"]["kmin"]

    # (1) linear scale ------------------------------------------------------
    ax = axes[0]
    kmax = int(k.max())
    width = max(kmax // 40, 1)
    bins = np.arange(-0.5, kmax + width + 0.5, width)
    ax.hist(k, bins=bins, color=P.BLUE, alpha=0.9, label="observed")
    grid = np.arange(0, kmax + 1)
    ax.plot(grid, stats.binom.pmf(grid, n - 1, res["p_er"]) * n * width, color=P.INK, lw=1.2, ls="--",
            label="random graph, same $N$ and $M$")
    ax.set_xlabel("degree $k$")
    ax.set_ylabel("number of stocks")
    ax.set_title(f"$r={r:.2f}$: linear scale")
    ax.legend(loc="upper right")
    ax.text(0.97, 0.60, f"{res['summary']['isolated']} isolated stocks ($k=0$)\n"
                         f"mean {k.mean():.1f}, median {k.median():.0f}, max {kmax}",
            transform=ax.transAxes, ha="right", fontsize=7.5, color=P.INK2)

    # (2) log-log P(k) ------------------------------------------------------
    ax = axes[1]
    pos = dist[dist["k"] > 0]
    ax.loglog(pos["k"], pos["pk"], "o", ms=3, color=P.NEUTRAL, label="raw $P(k)$")
    lb = log_binned_pdf(k.to_numpy())
    ax.loglog(lb["k_mid"], lb["density"], "s-", ms=4, lw=1.0, color=P.BLUE, label="log-binned")
    ax.set_xlabel("degree $k$")
    ax.set_ylabel("$P(k)$")
    ax.set_title(f"$r={r:.2f}$: log-log scale")
    ax.legend(loc="lower left")

    # (3) CCDF with fitted models ------------------------------------------
    ax = axes[2]
    ax.loglog(pos["k"], pos["ccdf"] * (k > 0).mean() / pos["ccdf"].iloc[0], "o", ms=3, color=P.BLUE,
              label="observed $P(K\\geq k)$")
    ks = np.arange(kmin, kmax + 1)
    scale = float((k >= kmin).mean())
    for name, fit in res["tail_fits"].items():
        col, ls = MODEL_STYLE[name]
        ax.loglog(ks, F.tail_ccdf(fit, ks) * scale, color=col, ls=ls, lw=1.4, label=name)
    ax.axvline(kmin, color=P.MUTED, lw=0.8, ls=":")
    ax.set_ylim(0.5 / n, 1.6)
    ax.text(kmin * 0.93, 1.05, f"$k_{{min}}={kmin}$", fontsize=7, color=P.INK2, ha="right")
    ax.set_xlabel("degree $k$")
    ax.set_ylabel("$P(K \\geq k)$")
    ax.set_title(f"$r={r:.2f}$: tail fits (MLE)")
    ax.legend(loc="lower left", fontsize=7)


def main() -> None:
    cfg = load_config()
    P.use_style()
    d = prepare(cfg, save=False)
    seed = cfg["nulls"]["seed"]
    r_star, r_back = cfg["network"]["baseline_threshold"], cfg["network"]["backbone_threshold"]

    results = {r: analyse(d.graph(r), seed, n_boot=500) for r in (r_star, r_back)}

    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.6))
    for row, r in enumerate((r_star, r_back)):
        draw_row(axes[row], results[r], r)
        for col in range(3):
            P.panel_label(axes[row, col], f"({'abcdef'[row * 3 + col]})")
    fig.tight_layout()
    P.savefig(fig, cfg, "fig03_degree_distribution")

    numbers = {}
    for r, tag in ((r_star, "star"), (r_back, "back")):
        res = results[r]
        res["k"].rename_axis("Ticker").to_frame().join(d.members[["Company Name", "sector", "psu"]]).sort_values(
            "degree", ascending=False).to_csv(Path(cfg["output"]["tables_dir"]) / f"A_degrees_r{r:.2f}.csv")
        save_table(degree_distribution(res["k"]), cfg, f"A_degree_distribution_r{r:.2f}", index=False, float_format="%.6f")
        save_table(res["tail_table"], cfg, f"A_tail_model_comparison_r{r:.2f}")
        save_table(res["full_table"], cfg, f"A_full_model_comparison_r{r:.2f}")
        for key, val in res["summary"].items():
            numbers[f"{tag}_{key}"] = val
        g = res["gof"]
        numbers.update({f"{tag}_kmin": g["kmin"], f"{tag}_alpha": g["alpha"], f"{tag}_ks": g["ks"],
                        f"{tag}_gof_p": g["p_value"], f"{tag}_ntail": g["n_tail"],
                        f"{tag}_tail_frac": g["n_tail"] / len(res["k"])})
        tt = res["tail_table"]
        numbers[f"{tag}_best_tail_model"] = tt["aic"].idxmin()
        numbers[f"{tag}_best_full_model"] = res["full_table"]["aic"].idxmin()
        for model, short in (("exponential", "exp"), ("log-normal", "ln"), ("power law + cut-off", "tpl")):
            numbers[f"{tag}_llr_{short}"] = tt.loc[model, "llr_powerlaw_vs_model"]
            numbers[f"{tag}_p_{short}"] = tt.loc[model, "vuong_p"]
            numbers[f"{tag}_daic_{short}"] = tt.loc[model, "delta_aic"]
            numbers[f"{tag}_full_daic_{short}"] = res["full_table"].loc[model, "delta_aic"]
        numbers[f"{tag}_daic_pl"] = tt.loc["power law", "delta_aic"]
        numbers[f"{tag}_full_daic_pl"] = res["full_table"].loc["power law", "delta_aic"]
        numbers[f"{tag}_full_alpha"] = res["full_fits"]["power law"].params["alpha"]
        numbers[f"{tag}_tpl_alpha"] = res["tail_fits"]["power law + cut-off"].params["alpha"]
        numbers[f"{tag}_tpl_lambda"] = res["tail_fits"]["power law + cut-off"].params["lambda"]
        numbers[f"{tag}_exp_lambda"] = res["tail_fits"]["exponential"].params["lambda"]
        numbers[f"{tag}_full_tpl_alpha"] = res["full_fits"]["power law + cut-off"].params["alpha"]
        numbers[f"{tag}_full_tpl_lambda"] = res["full_fits"]["power law + cut-off"].params["lambda"]
        numbers[f"{tag}_full_cutoff_scale"] = 1.0 / res["full_fits"]["power law + cut-off"].params["lambda"]
        numbers[f"{tag}_full_llr_tpl"] = res["full_table"].loc["power law + cut-off", "llr_powerlaw_vs_model"]
        numbers[f"{tag}_n_connected"] = int((res["k"] > 0).sum())
        print(f"\n=== r = {r:.2f} ===")
        print(pd.Series(res["summary"]).round(3).to_string())
        print("GoF:", g)
        print(tt.round(3).to_string())
        print("full range:\n", res["full_table"].round(3).to_string())

    # the same test across a ladder of thresholds
    rows = []
    for r in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60):
        k = degree_series(d.graph(r))
        pos = k[k > 0].to_numpy()
        gof = F.powerlaw_gof(pos, n_boot=300, seed=seed)
        tt = F.comparison_table(F.fit_tail(pos, gof["kmin"]))
        rows.append({"threshold": r, "nodes_k>0": len(pos), "mean_degree": k.mean(), "max_degree": int(k.max()),
                     "kmin": gof["kmin"], "n_tail": gof["n_tail"], "alpha": gof["alpha"], "ks": gof["ks"],
                     "gof_p": gof["p_value"], "best_model_aic": tt["aic"].idxmin(),
                     "vuong_p_vs_exp": tt.loc["exponential", "vuong_p"],
                     "llr_vs_exp": tt.loc["exponential", "llr_powerlaw_vs_model"],
                     "vuong_p_vs_cutoff": tt.loc["power law + cut-off", "vuong_p"],
                     "llr_vs_cutoff": tt.loc["power law + cut-off", "llr_powerlaw_vs_model"]})
    ladder = pd.DataFrame(rows).set_index("threshold")
    save_table(ladder, cfg, "A_powerlaw_test_by_threshold")
    print("\n", ladder.round(3).to_string())
    numbers["ladder_n_pl_plausible"] = int((ladder["gof_p"] >= 0.1).sum())
    numbers["ladder_n"] = len(ladder)
    save_numbers("A", numbers)


if __name__ == "__main__":
    main()
