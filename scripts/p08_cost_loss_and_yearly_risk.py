"""Step 8. Decision value (cost/loss) and yearly district risk ranking.

1) Relative economic value (Richardson 2000). For a cost/loss ratio a = C/L, where C is the
   cost of acting in a district for a week and L the loss that action avoids:
     E_never   = min(a, s)                      (always act or never act, whichever is cheaper)
     E_perfect = s * a
     E_model   = F (1 - s) a + H s a + (1 - H) s
     V         = (E_never - E_model) / (E_never - E_perfect)
   with H = POD, F = POFD and s = outbreak base rate. V = 1 is a perfect forecast, V <= 0 is no
   better than the simplest fixed strategy. Shown for (i) the best threshold per a (potential
   value) and (ii) the single POD-oriented threshold chosen out-of-fold (what a fixed trigger gets).
   Predictions of the four test years 2020-2023 are pooled.

2) Yearly district ranking: for each test year, Spearman correlation between each district's
   mean predicted risk and its number of outbreak weeks, and overlap of the top-10 districts.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, warnings
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
from scipy.stats import spearmanr
warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "font.size": 10})

CONFIGS = [("Climate district", "LR", "#2a78d6", "-"),
           ("Case history only (AR)", "LR", "#eb6834", "-"),
           ("Climate district + case history", "LR", "#1baf7a", "-")]
ALPHAS = np.round(np.logspace(np.log10(0.005), np.log10(0.5), 40), 4)


def value(H, F, s, a):
    e_never, e_perf = np.minimum(a, s), s * a
    e_mod = F * (1 - s) * a + H * s * a + (1 - H) * s
    return (e_never - e_mod) / (e_never - e_perf)


def hf_curve(y, p):
    order = np.argsort(-p); ys = y[order]
    tp, fp = np.cumsum(ys), np.cumsum(1 - ys)
    return tp / ys.sum(), fp / (len(ys) - ys.sum())


def main():
    pred = pd.read_csv(OUT / "04_rolling_origin_predictions.csv.gz")
    pred = pred[(pred.setting == "all districts") & (pred.test_year <= 2023)]
    res = pd.read_csv(OUT / "04_rolling_origin_by_year.csv")
    res = res[(res.setting == "all districts") & (res.threshold_rule == "POD>=0.8") & (res.test_year <= 2023)]
    rows, curves = [], {}
    for fs, model, *_ in CONFIGS:
        g = pred[(pred.features == fs) & (pred.model == model)]
        y, p = g.y.values, g.p.values; s = y.mean()
        H, F = hf_curve(y, p)
        # fixed trigger: the POD-oriented threshold of each fold, applied to that fold's year
        r = res[(res.features == fs) & (res.model == model)]
        tp, fp, fn = r.TP.sum(), r.FP.sum(), r.FN.sum(); tn = (r.n_test - r.pos_test).sum() - fp
        Hf, Ff = tp / (tp + fn), fp / (fp + tn)
        best, fixed = [], []
        for a in ALPHAS:
            best.append(np.max(value(H, F, s, a))); fixed.append(value(Hf, Ff, s, a))
            rows.append(dict(features=fs, model=model, cost_loss=a, value_best_threshold=best[-1],
                             value_fixed_trigger=fixed[-1]))
        curves[(fs, model)] = (best, fixed)
    v = pd.DataFrame(rows); v.round(4).to_csv(OUT / "08_relative_economic_value.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    name = {"Climate district": "District climate", "Case history only (AR)": "Case history",
            "Climate district + case history": "Case history + district climate"}
    for fs, model, c, ls in CONFIGS:
        best, fixed = curves[(fs, model)]
        ax.plot(ALPHAS, np.clip(best, -0.2, 1), color=c, ls=ls, lw=2, label=f"{name[fs]} ({model})")
        ax.plot(ALPHAS, np.clip(fixed, -0.2, 1), color=c, ls=":", lw=1.2)
    ax.set_xscale("log"); ax.set_ylim(-0.2, 1); ax.axhline(0, color="#52514e", lw=0.8)
    ticks = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FixedFormatter([str(t) for t in ticks]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("Cost/loss ratio of acting (C/L)"); ax.set_ylabel("Relative economic value")
    ax.grid(color="#e4e3df", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Value of the forecasts for decisions (solid: best threshold, dotted: fixed trigger)",
                 fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    fig.tight_layout(); fig.savefig(OUT / "fig_relative_economic_value.png", dpi=300); plt.close(fig)

    # yearly ranking of districts
    yr = []
    for (fs, model), g in pred.groupby(["features", "model"]):
        for Y, gy in g.groupby("test_year"):
            d = gy.groupby("district").agg(risk=("p", "mean"), obs=("y", "sum"))
            if d.obs.sum() == 0: continue
            top_m, top_o = set(d.risk.nlargest(10).index), set(d.obs.nlargest(10).index)
            yr.append(dict(features=fs, model=model, test_year=Y, spearman=spearmanr(d.risk, d.obs).correlation,
                           top10_overlap=len(top_m & top_o), districts_with_outbreaks=int((d.obs > 0).sum())))
    yr = pd.DataFrame(yr); yr.round(3).to_csv(OUT / "08_yearly_district_ranking.csv", index=False)

    pick = [(a, b) for a, b, *_ in CONFIGS]
    print("Relative economic value at selected cost/loss ratios (best threshold / fixed trigger):")
    for a in [0.01, 0.05, 0.1, 0.2]:
        i = int(np.argmin(abs(ALPHAS - a)))
        print(f"  C/L={ALPHAS[i]:.3f}: " + " | ".join(f"{fs[:22]} {m}: {curves[(fs, m)][0][i]:.2f}/{curves[(fs, m)][1][i]:.2f}"
                                                    for fs, m in pick))
    s = yr[yr.set_index(["features", "model"]).index.isin(pick)].groupby(["features", "model"])[["spearman", "top10_overlap"]].mean()
    print("\nYearly district ranking (mean over 2020-2023):\n" + s.round(2).to_string())


if __name__ == "__main__":
    main()
