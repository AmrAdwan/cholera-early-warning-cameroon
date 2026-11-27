"""Step 7. One compact results table (POD-oriented threshold, 2020-2023 mean +/- SD)."""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import pandas as pd

ROWS = [("Climate district", "RF"), ("Climate district", "LR"),
        ("Climate district, operational lags", "RF"), ("Persistence", "Persistence"),
        ("Case history only (AR)", "RF"), ("Case history only (AR)", "LR"),
        ("Case history, 1-wk reporting delay", "LR"),
        ("Climate district + case history", "RF"), ("Climate district + case history", "LR"),
        ("Climate district + case history", "XGB"),
        ("Climate district op. lags + case history", "RF"), ("Climate district op. lags + case history", "LR"),
        ("Climate district op. lags + delayed case history", "LR"),
        ("Climate district + WASH lagged", "RF")]


def main():
    s = pd.read_csv(OUT / "04_rolling_origin_summary.csv")
    s = s[(s.setting == "all districts") & (s.threshold_rule == "POD>=0.8")].set_index(["features", "model"])
    o = pd.read_csv(OUT / "06_onset_summary.csv")
    o = o[o.setting == "all districts"].set_index(["features", "model"])
    f = lambda m, sd: f"{m:.2f} +/- {sd:.2f}"
    out = []
    for k in ROWS:
        r = s.loc[k]; on = o.loc[k] if k in o.index else None
        out.append({"Features": k[0], "Model": k[1], "AUC": f(r.AUC_mean, r.AUC_std), "PR-AUC": f"{r.PR_AUC_mean:.2f}",
                    "POD": f(r.POD_mean, r.POD_std), "FAR": f"{r.FAR_mean:.2f}", "F1": f"{r.F1_mean:.2f}",
                    "False alarms 2024": int(r.FP_2024), "Onset AUC": f"{on.onset_AUC_mean:.2f}" if on is not None else "",
                    "Onset POD": f"{on.onset_POD_mean:.2f}" if on is not None else ""})
    t = pd.DataFrame(out); t.to_csv(OUT / "07_main_results_table.csv", index=False)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
