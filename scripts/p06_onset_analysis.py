"""Step 6. Can the models anticipate the START of an outbreak?

Case history scores well partly because outbreaks last several weeks: once a district
is in an outbreak, last week's cases predict this week well. For anticipatory action
the harder question is the first week of an outbreak. Here 'onset' = outbreak week
whose previous week in the same district was not an outbreak week. AUC is computed
for onset weeks vs. non-outbreak weeks (ongoing outbreak weeks are left out), and
POD for onsets uses the same POD-oriented threshold as the main evaluation.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    df = pd.read_csv(OUT / "features_weekly.csv")[["district", "year", "week", "outbreak"]].sort_values(["district", "year", "week"])
    df["prev"] = df.groupby("district").outbreak.shift(1).fillna(0)
    df["onset"] = ((df.outbreak == 1) & (df.prev == 0)).astype(int)
    pred = pd.read_csv(OUT / "04_rolling_origin_predictions.csv.gz").merge(df[["district", "year", "week", "onset"]],
                                                                           on=["district", "year", "week"])
    res = pd.read_csv(OUT / "04_rolling_origin_by_year.csv")
    thr = res[res.threshold_rule == "POD>=0.8"].set_index(["setting", "test_year", "features", "model"]).threshold
    rows = []
    for k, g in pred[pred.test_year <= 2023].groupby(["setting", "test_year", "features", "model"]):
        sub = g[(g.onset == 1) | (g.y == 0)]
        if sub.onset.sum() == 0: continue
        t = thr.loc[k]
        rows.append(dict(zip(["setting", "test_year", "features", "model"], k), n_onsets=int(sub.onset.sum()),
                         onset_AUC=roc_auc_score(sub.onset, sub.p), onset_POD=float((g[g.onset == 1].p >= t).mean())))
    r = pd.DataFrame(rows); r.to_csv(OUT / "06_onset_by_year.csv", index=False)
    s = r.groupby(["setting", "features", "model"]).agg(n_onsets=("n_onsets", "sum"), onset_AUC_mean=("onset_AUC", "mean"),
                                                         onset_AUC_sd=("onset_AUC", "std"), onset_POD_mean=("onset_POD", "mean")).round(3)
    s.reset_index().to_csv(OUT / "06_onset_summary.csv", index=False)
    print(s.sort_values("onset_AUC_mean", ascending=False).to_string())


if __name__ == "__main__":
    main()
