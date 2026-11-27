"""Step 9. Sensitivity checks for information from the whole period
(hyperparameters and outbreak thresholds).

A) Nested tuning. Instead of fixed hyperparameters, each fold tunes the Random Forest
   (max_depth x min_samples_leaf) on its own training years only, using the same
   year-by-year out-of-fold scheme. Reports the chosen settings, the inner CV AUC
   (model-development metric) and the test AUC, next to the fixed-parameter results.

B) Outbreak threshold from training years only. The Zheng et al. (2022) label uses each
   district's mean weekly cases over 2018-2024, i.e. including the test years. Here, for
   each test year Y, the threshold is the district mean over the years before Y, and all
   labels (training and test) are recomputed with it. Districts without any earlier case
   get a minimal threshold (any week with a case can start an outbreak), otherwise they
   could never be labelled. Reports label agreement and model performance.
"""
import sys, warnings; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import p04_evaluate as E
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
warnings.filterwarnings("ignore")

YEARS = [2020, 2021, 2022, 2023, 2024]
GRID = [dict(max_depth=d, min_samples_leaf=l) for d in (5, 10, 20, None) for l in (1, 5, 20)]


# ---------- A) nested tuning ----------
def rf(params, y):
    pos, neg = int(y.sum()), int((y == 0).sum())
    cw = {0: len(y) / (2 * neg), 1: len(y) / (2 * max(pos, 1))}
    return RandomForestClassifier(n_estimators=200, max_features="sqrt", min_samples_split=5,
                                  class_weight=cw, random_state=42, n_jobs=-1, **params)


def nested(df, fs):
    feats = E.FEATURE_SETS[fs]; rows = []
    for Y in YEARS:
        tr, te = df[df.year < Y], df[df.year == Y]
        best, best_auc, best_oof = None, -1, None
        for params in GRID:
            ys, ps = [], []
            for y_in in range(2019, Y):
                a, b = tr[tr.year < y_in], tr[tr.year == y_in]
                if a.outbreak.sum() == 0 or b.outbreak.sum() == 0: continue
                ps.append(rf(params, a.outbreak).fit(a[feats], a.outbreak).predict_proba(b[feats])[:, 1])
                ys.append(b.outbreak.values)
            yy, pp = np.concatenate(ys), np.concatenate(ps)
            auc = roc_auc_score(yy, pp)
            if auc > best_auc: best, best_auc, best_oof = params, auc, (yy, pp)
        thr = E.thresholds(*best_oof)["POD>=0.8"]
        p = rf(best, tr.outbreak).fit(tr[feats], tr.outbreak).predict_proba(te[feats])[:, 1]
        y = te.outbreak.values
        rows.append(dict(check="nested tuning", features=fs, model="RF (tuned per fold)", test_year=Y,
                         chosen=str(best), inner_cv_AUC=best_auc,
                         AUC=roc_auc_score(y, p) if 0 < y.sum() < len(y) else np.nan, **E.scores(y, p, thr)))
        print(f"  [nested] {Y} {fs[:35]:35s} {best} innerAUC={best_auc:.3f} AUC={rows[-1]['AUC']:.3f}", flush=True)
    return rows


# ---------- B) training-only outbreak threshold ----------
def zheng_with_threshold(cases, threshold):
    """Zheng et al. labelling (same rules as p01) with a given threshold."""
    n = len(cases); lab = np.zeros(n, dtype=int); i = 0
    while i < n:
        if cases[i] >= threshold and (i + 2) < n and cases[i + 1] > cases[i] and cases[i + 2] > cases[i + 1]:
            start, j, end = i, i + 3, i + 2
            while j < n:
                if j + 1 < n and cases[j] < threshold and cases[j + 1] < threshold:
                    if all(cases[w] < threshold for w in range(j + 2, min(j + 6, n))):
                        end = j - 1; break
                    j += 1
                else:
                    end = j; j += 1
            if j >= n: end = n - 1
            lab[start:end + 1] = 1; i = end + 1; continue
        i += 1
    return lab


def relabel(df, Y):
    out = df.sort_values(["district", "year", "week"]).copy(); new = []
    for d, g in out.groupby("district", sort=False):
        c = g.cases.fillna(0).values
        thr = g.loc[g.year < Y, "cases"].fillna(0).mean()
        new.append(zheng_with_threshold(c, thr if thr > 0 else 1e-9))  # no earlier cases: any case counts
    out["outbreak"] = np.concatenate(new)
    return out


def train_only_labels(df):
    cfg = [("Climate district", "LR"), ("Case history only (AR)", "LR"), ("Climate district + case history", "LR"),
           ("Climate district + case history", "RF")]
    rows, agree = [], []
    for Y in YEARS:
        d = relabel(df, Y)
        t = d[d.year == Y]; t0 = df.set_index(["district", "year", "week"]).loc[t.set_index(["district", "year", "week"]).index]
        agree.append(dict(test_year=Y, outbreak_weeks_whole_period=int(t0.outbreak.sum()),
                          outbreak_weeks_train_only=int(t.outbreak.sum()),
                          agreement=float((t0.outbreak.values == t.outbreak.values).mean())))
        r, _ = E.run(d, cfg, [Y], tag="train-only threshold")
        rows.append(r)
    return pd.concat(rows), pd.DataFrame(agree)


def main():
    df = pd.read_csv(OUT / "features_weekly.csv")
    A = pd.DataFrame(nested(df, "Climate district") + nested(df, "Climate district + case history"))
    A.to_csv(OUT / "09a_nested_tuning_by_year.csv", index=False)
    B, agree = train_only_labels(df)
    B.to_csv(OUT / "09b_train_only_threshold_by_year.csv", index=False); agree.to_csv(OUT / "09b_label_agreement.csv", index=False)

    base = pd.read_csv(OUT / "04_rolling_origin_by_year.csv")
    base = base[(base.setting == "all districts") & (base.threshold_rule == "POD>=0.8")]
    ev = lambda x: x[x.test_year <= 2023]
    comp = []
    for fs in ["Climate district", "Climate district + case history"]:
        b = ev(base[(base.features == fs) & (base.model == "RF")]); a = ev(A[A.features == fs])
        comp += [dict(check="A nested tuning", features=fs, model="RF fixed params", AUC=b.AUC.mean(), POD=b.POD.mean(), FAR=b.FAR.mean()),
                 dict(check="A nested tuning", features=fs, model="RF tuned per fold", AUC=a.AUC.mean(), POD=a.POD.mean(), FAR=a.FAR.mean(),
                      inner_cv_AUC=a.inner_cv_AUC.mean())]
    Bp = ev(B[B.threshold_rule == "POD>=0.8"])
    for (fs, m), g in Bp.groupby(["features", "model"]):
        b = ev(base[(base.features == fs) & (base.model == m)])
        comp += [dict(check="B outbreak threshold", features=fs, model=f"{m}, whole-period threshold", AUC=b.AUC.mean(), POD=b.POD.mean(), FAR=b.FAR.mean()),
                 dict(check="B outbreak threshold", features=fs, model=f"{m}, training-years threshold", AUC=g.AUC.mean(), POD=g.POD.mean(), FAR=g.FAR.mean())]
    comp = pd.DataFrame(comp).round(3); comp.to_csv(OUT / "09_sensitivity_summary.csv", index=False)
    print(comp.to_string(index=False)); print(agree.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
