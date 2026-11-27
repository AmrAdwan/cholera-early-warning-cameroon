"""Step 4. Rolling-origin evaluation.

Expanding window: for each test year Y in 2020-2024, train on 2018..Y-1, test on Y.
Results are reported per year and as mean +/- SD over 2020-2023 (2024 has no
outbreak weeks, so only false alarms can be counted there).

Nothing is fitted on the whole period: hyperparameters are fixed in advance,
class weights come from each training window, and the decision threshold is chosen
on out-of-fold predictions inside the training window (each training year predicted
by a model fitted on the years before it), not on in-sample predictions.

Threshold rules: Youden's J, max F1, and an operational rule that keeps the
probability of detection (POD) >= 0.8 on the out-of-fold data.
Reported: AUC, PR-AUC, POD (= recall), FAR (false alarm ratio, FP/alerts),
POFD (false alarm rate, FP/non-events), precision, F1.
A second run keeps only districts with an outbreak in the training years
(structural zeros removed).
"""
import sys, time, warnings; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
warnings.filterwarnings("ignore")

L12 = lambda c: [f"{c}_lag1", f"{c}_lag2"]
CLIM_D = L12("mean_temp") + L12("mean_dewpoint") + L12("total_precip") + ["max_precip_lag1"]
CLIM_D_OP = [f"{c}_lag{l}" for c in ["mean_temp", "mean_dewpoint", "total_precip"] for l in (2, 3)] + ["max_precip_lag2"]
AR = ["cases_lag1", "cases_lag2", "cases_4wk_lag1"]
AR_DELAY = ["cases_lag2", "cases_lag3", "cases_4wk_lag2"]  # one week of reporting delay
RFC = ["rf_water_count", "rf_sanitation_count", "rf_hygiene_count", "rf_other_count"]
WASH_LAG = [c + "_lag1" for c in RFC]

FEATURE_SETS = {
    "Climate district": CLIM_D,
    "Climate district, operational lags": CLIM_D_OP,
    "Case history only (AR)": AR,
    "Climate district + case history": CLIM_D + AR,
    "Climate district op. lags + case history": CLIM_D_OP + AR,
    "Case history, 1-wk reporting delay": AR_DELAY,
    "Climate district op. lags + delayed case history": CLIM_D_OP + AR_DELAY,
    "Climate district + WASH lagged": CLIM_D + WASH_LAG,
}
TREES_ALL = {"Climate district", "Climate district + case history", "Climate district op. lags + case history"}


def make_model(name, y):
    pos, neg = int(y.sum()), int((y == 0).sum()); spw = neg / max(pos, 1)
    cw = {0: len(y) / (2 * neg), 1: len(y) / (2 * max(pos, 1))}
    if name == "RF":
        return RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=5, min_samples_leaf=5,
                                      max_features="sqrt", class_weight=cw, random_state=42, n_jobs=-1)
    if name == "XGB":
        return XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, scale_pos_weight=spw, eval_metric="logloss", random_state=42, n_jobs=-1)
    if name == "LGBM":
        return LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                              colsample_bytree=0.8, scale_pos_weight=spw, random_state=42, n_jobs=-1, verbose=-1)
    if name == "LR":
        return make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    raise ValueError(name)


def fit_predict(model, df_tr, df_te, feats):
    if model == "Persistence":  # benchmark: last week's (log) case count is the risk score
        return df_te["cases_lag1"].values
    m = make_model(model, df_tr.outbreak).fit(df_tr[feats], df_tr.outbreak)
    return m.predict_proba(df_te[feats])[:, 1]


def thresholds(y, p):
    out = {}
    fpr, tpr, th = roc_curve(y, p)
    out["Youden"] = float(th[np.argmax(tpr - fpr)])
    order = np.argsort(-p); ys = y[order]; tp = np.cumsum(ys); k = np.arange(1, len(ys) + 1)
    f1 = 2 * tp / (k + ys.sum()); out["MaxF1"] = float(p[order][np.argmax(f1)])
    rec = tp / max(ys.sum(), 1); i = np.argmax(rec >= 0.8); out["POD>=0.8"] = float(p[order][i])
    return out


def scores(y, p, thr):
    a = p >= thr; tp = int((a & (y == 1)).sum()); fp = int((a & (y == 0)).sum())
    fn = int((~a & (y == 1)).sum()); tn = int((~a & (y == 0)).sum())
    pod = tp / (tp + fn) if tp + fn else np.nan; far = fp / (tp + fp) if tp + fp else np.nan
    prec = 1 - far if tp + fp else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if tp + fn else np.nan
    return dict(POD=pod, FAR=far, POFD=fp / (fp + tn), precision=prec, F1=f1, alerts=tp + fp, TP=tp, FP=fp, FN=fn)


def run(df, configs, test_years, at_risk_only=False, tag=""):
    rows, preds = [], []
    for Y in test_years:
        tr_all, te_all = df[df.year < Y], df[df.year == Y]
        if at_risk_only:  # structural zeros: keep districts with >=1 outbreak week in the TRAINING years only
            keep = tr_all.groupby("district").outbreak.sum().loc[lambda s: s > 0].index
            tr_all, te_all = tr_all[tr_all.district.isin(keep)], te_all[te_all.district.isin(keep)]
        for fs, model in configs:
            feats = FEATURE_SETS.get(fs, [])
            # out-of-fold predictions inside the training window, for choosing the threshold
            oof_y, oof_p = [], []
            for y_in in range(2019, Y):
                a, b = tr_all[tr_all.year < y_in], tr_all[tr_all.year == y_in]
                if a.outbreak.sum() == 0 or b.outbreak.sum() == 0:
                    continue
                oof_p.append(fit_predict(model, a, b, feats)); oof_y.append(b.outbreak.values)
            thr = thresholds(np.concatenate(oof_y), np.concatenate(oof_p))
            p = fit_predict(model, tr_all, te_all, feats); y = te_all.outbreak.values
            base = dict(setting=tag, test_year=Y, features=fs, model=model, n_test=len(y), pos_test=int(y.sum()),
                        AUC=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan,
                        PR_AUC=average_precision_score(y, p) if y.sum() > 0 else np.nan)
            for rule, t in thr.items():
                rows.append({**base, "threshold_rule": rule, "threshold": t, **scores(y, p, t)})
            preds.append(pd.DataFrame({"setting": tag, "test_year": Y, "features": fs, "model": model,
                                       "district": te_all.district.values, "year": te_all.year.values,
                                       "week": te_all.week.values, "y": y, "p": p}))
            print(f"  [{tag}] {Y} {model:11s} {fs[:45]:45s} AUC={base['AUC']:.3f}", flush=True)
    return pd.DataFrame(rows), pd.concat(preds)


def main():
    t0 = time.time()
    df = pd.read_csv(OUT / "features_weekly.csv")
    configs = [(fs, m) for fs in FEATURE_SETS for m in ["RF", "LR"]]
    configs += [(fs, m) for fs in TREES_ALL for m in ["XGB", "LGBM"]]
    configs += [("Persistence", "Persistence")]
    years = [2020, 2021, 2022, 2023, 2024]
    res, pred = run(df, configs, years, tag="all districts")
    sz_cfg = [(fs, "RF") for fs in ["Climate district", "Case history only (AR)",
                                    "Climate district + case history", "Climate district op. lags + case history"]]
    sz_cfg += [("Climate district op. lags + delayed case history", "RF"), ("Persistence", "Persistence")]
    res2, pred2 = run(df, sz_cfg, years, at_risk_only=True, tag="at-risk districts only")
    res = pd.concat([res, res2]); pred = pd.concat([pred, pred2])
    res.to_csv(OUT / "04_rolling_origin_by_year.csv", index=False)
    pred.to_csv(OUT / "04_rolling_origin_predictions.csv.gz", index=False)

    ev = res[res.test_year <= 2023]
    keys = ["setting", "features", "model", "threshold_rule"]
    summ = ev.groupby(keys)[["AUC", "PR_AUC", "POD", "FAR", "POFD", "F1"]].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    quiet = res[res.test_year == 2024].groupby(keys)[["FP", "POFD"]].first().add_suffix("_2024")
    summ = summ.join(quiet).reset_index().round(3)
    summ.to_csv(OUT / "04_rolling_origin_summary.csv", index=False)
    print(f"done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
