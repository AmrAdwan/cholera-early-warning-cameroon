"""Step 11. Do the WASH risk-factor counts add information when they are lagged?

The counts are the number of patients that week who reported a risk factor, so they
are only non-zero in weeks with cases. Lagged by one week they are legitimate
predictors (known at forecast time), but they are then a subset of last week's cases.
The question is whether they add anything beyond case history. Same rolling-origin
setup and threshold rule as step 4.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import p04_evaluate as E
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    df = pd.read_csv(OUT / "features_weekly.csv")
    E.FEATURE_SETS["WASH lagged only"] = E.WASH_LAG
    E.FEATURE_SETS["Case history + WASH lagged"] = E.AR + E.WASH_LAG
    E.FEATURE_SETS["Climate district + case history + WASH lagged"] = E.CLIM_D + E.AR + E.WASH_LAG
    sets = ["WASH lagged only", "Climate district",
            "Climate district + WASH lagged", "Case history only (AR)", "Case history + WASH lagged",
            "Climate district + case history", "Climate district + case history + WASH lagged"]
    cfg = [(fs, m) for fs in sets for m in ["RF", "LR"]]
    res, pred = E.run(df, cfg, [2020, 2021, 2022, 2023, 2024], tag="WASH")
    res.to_csv(OUT / "11_wash_lagged_by_year.csv", index=False)
    r = res[res.threshold_rule == "POD>=0.8"]
    s = r[r.test_year <= 2023].groupby(["features", "model"])[["AUC", "POD", "FAR"]].mean()
    s["FP_2024"] = r[r.test_year == 2024].set_index(["features", "model"]).FP

    # onset AUC (first outbreak week vs non-outbreak weeks), as in step 6
    o = df[["district", "year", "week", "outbreak"]].sort_values(["district", "year", "week"])
    o["onset"] = ((o.outbreak == 1) & (o.groupby("district").outbreak.shift(1).fillna(0) == 0)).astype(int)
    pred = pd.DataFrame(pred) if not isinstance(pred, pd.DataFrame) else pred
    pred = pred.merge(o[["district", "year", "week", "onset"]], on=["district", "year", "week"])
    on = []
    for (Y, fs, m), g in pred[pred.test_year <= 2023].groupby(["test_year", "features", "model"]):
        sub = g[(g.onset == 1) | (g.y == 0)]
        if sub.onset.sum(): on.append(dict(features=fs, model=m, onset_AUC=roc_auc_score(sub.onset, sub.p)))
    s["onset_AUC"] = pd.DataFrame(on).groupby(["features", "model"]).onset_AUC.mean()
    s = s.reindex([(fs, m) for fs in sets for m in ["RF", "LR"]])
    s.round(3).to_csv(OUT / "11_wash_lagged_summary.csv"); print(s.round(3).to_string())


if __name__ == "__main__":
    main()
