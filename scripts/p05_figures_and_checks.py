"""Step 5. Multicollinearity check and figures.

- VIF + correlation matrix for the predictors
- Fig: AUC per test year (rolling origin) for the main feature sets
- Fig: permutation feature importance (Random Forest, test year 2023)
- Fig: observed outbreaks and model alerts in 2023, two stacked panels
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import p04_evaluate as E
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.inspection import permutation_importance

BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8c8b85"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.family": "serif", "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})
LABEL = {"mean_temp": "Temperature", "mean_dewpoint": "Dewpoint", "total_precip": "Rainfall, weekly total",
         "max_precip": "Rainfall, daily maximum", "total_runoff": "Surface runoff", "cases": "Cases"}

def pretty(f):
    if f == "cases_4wk_lag1": return "Cases, previous 4 weeks"
    for k, v in LABEL.items():
        if f.startswith(k):
            lag = f.split("_lag")[-1]; return f"{v} ({'1 week' if lag == '1' else lag + ' weeks'} before)"
    return f


def vif_table(df):
    feats = E.CLIM_D + E.AR
    X = df[df.year <= 2023][feats].dropna(); X = (X - X.mean()) / X.std(); X["const"] = 1.0
    vif = pd.DataFrame({"feature": feats, "VIF": [variance_inflation_factor(X.values, i) for i in range(len(feats))]})
    vif.round(2).to_csv(OUT / "05_vif.csv", index=False)
    df[feats].corr().round(2).to_csv(OUT / "05_correlation_matrix.csv")
    return vif


def fig_auc_by_year(res):
    sel = {"Climate district": (BLUE, "-"), "Case history only (AR)": (ORANGE, "-"),
           "Climate district + case history": (AQUA, "-")}
    name = {"Climate district": "District climate", "Case history only (AR)": "Case history",
            "Climate district + case history": "Case history + district climate"}
    r = res[(res.setting == "all districts") & (res.model == "RF") & (res.threshold_rule == "Youden")
            & (res.test_year <= 2023) & res.features.isin(sel)]
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    for fs, (c, ls) in sel.items():
        d = r[r.features == fs].sort_values("test_year")
        ax.plot(d.test_year, d.AUC, color=c, ls=ls, lw=2, marker="o", ms=6, label=name[fs])
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":"); ax.text(2019.93, 0.515, "chance", va="bottom", fontsize=8, color=INK2)
    pos = res[(res.test_year <= 2023)].groupby("test_year").pos_test.first()
    ax.set_xticks(pos.index); ax.set_xticklabels([f"{y}\n({n} outbreak wks)" for y, n in pos.items()], fontsize=8)
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("AUC on test year"); ax.grid(axis="y", color=GRID, lw=0.6)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_xlim(2019.8, 2023.2)
    ax.set_title("Random Forest, rolling-origin evaluation", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_auc_by_test_year.png", dpi=300); plt.close(fig)


def fig_importance(df):
    feats = E.CLIM_D + E.AR
    tr, te = df[df.year <= 2022], df[df.year == 2023]
    m = E.make_model("RF", tr.outbreak).fit(tr[feats], tr.outbreak)
    pi = permutation_importance(m, te[feats], te.outbreak, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
    imp = pd.DataFrame({"feature": feats, "drop_in_AUC": pi.importances_mean, "sd": pi.importances_std}).sort_values("drop_in_AUC")
    imp.round(4).to_csv(OUT / "05_permutation_importance_2023.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.barh([pretty(f) for f in imp.feature], imp.drop_in_AUC, xerr=imp.sd, color=BLUE, height=0.6,
            error_kw=dict(ecolor=INK2, lw=0.8, capsize=2))
    ax.axvline(0, color=INK2, lw=0.8); ax.tick_params(axis="y", labelsize=10)
    ax.set_xlabel("Drop in AUC when shuffled (test year 2023)")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title("Random Forest, district climate + case history", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_permutation_importance.png", dpi=300); plt.close(fig)
    return imp


def fig_risk_over_time(pred, res):
    fs = "Climate district + case history"
    p = pred[(pred.setting == "all districts") & (pred.model == "RF") & (pred.features == fs) & (pred.test_year == 2023)].copy()
    thr = res[(res.setting == "all districts") & (res.model == "RF") & (res.features == fs) & (res.test_year == 2023)
              & (res.threshold_rule == "POD>=0.8")].threshold.iloc[0]
    p["alert"] = p.p >= thr
    wk = p.groupby("week").agg(observed=("y", "sum"), alerts=("alert", "sum"),
                               hits=("y", lambda s: int((s & p.loc[s.index, "alert"]).sum())))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
    a1.bar(wk.index, wk.observed, color=INK2, width=0.8); a1.set_ylabel("Districts in\noutbreak")
    a2.bar(wk.index, wk.hits, color=AQUA, width=0.8, label="correct alert (district in outbreak)")
    a2.bar(wk.index, wk.alerts - wk.hits, bottom=wk.hits, color=BLUE, width=0.8, label="false alarm")
    a2.set_ylabel("Districts\nalerted"); a2.set_xlabel("Epidemiological week, 2023")
    a2.legend(frameon=False, fontsize=8, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1)
    for a in (a1, a2): a.grid(axis="y", color=GRID, lw=0.6)
    a1.set_title("Observed outbreaks and model alerts, 2023", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(OUT / "fig_observed_vs_alerts_2023.png", dpi=300); plt.close(fig)


def main():
    df = pd.read_csv(OUT / "features_weekly.csv")
    res = pd.read_csv(OUT / "04_rolling_origin_by_year.csv")
    pred = pd.read_csv(OUT / "04_rolling_origin_predictions.csv.gz")
    print(vif_table(df).round(2).to_string(index=False))
    fig_auc_by_year(res); print(fig_importance(df).round(4).to_string(index=False)); fig_risk_over_time(pred, res)
    print("figures written")


if __name__ == "__main__":
    main()
