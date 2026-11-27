"""Step 1. Build the district-week panel.

Spelling variants of the same health district ('LIMBE'/'Limbe', 'SOA'/'Soa'/'soa', ...)
are merged first and their case counts summed. Outbreak weeks are then labelled with
the definition of Zheng et al. (2022): per district, the threshold is the mean weekly
case count; an outbreak starts at a week at or above the threshold followed by two
weeks of increasing counts, and ends before two consecutive weeks below the threshold
that are followed by four further weeks without exceeding it.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, pandas as pd


def label_outbreaks_zheng(weekly_full):  # outbreak definition of Zheng et al. (2022)
    weekly_full = weekly_full.sort_values(["district", "year", "week"]).copy()
    weekly_full["outbreak"] = 0
    for district, grp in weekly_full.groupby("district"):
        idx = grp.index.tolist(); cases = grp["cases"].values; n = len(cases)
        threshold = cases.mean()
        if threshold == 0:
            continue
        i = 0
        while i < n:
            if cases[i] >= threshold and (i + 2) < n:
                if cases[i + 1] > cases[i] and cases[i + 2] > cases[i + 1]:
                    start = i; j = i + 3; end = j - 1
                    while j < n:
                        if j + 1 < n and cases[j] < threshold and cases[j + 1] < threshold:
                            washout_ok = True
                            for w in range(j + 2, min(j + 6, n)):
                                if cases[w] >= threshold:
                                    washout_ok = False; break
                            if washout_ok:
                                end = j - 1; break
                            else:
                                j += 1
                        else:
                            end = j; j += 1
                    if j >= n:
                        end = n - 1
                    for k in range(start, end + 1):
                        weekly_full.loc[idx[k], "outbreak"] = 1
                    i = end + 1
                    continue
            i += 1
    return weekly_full


def main():
    w = pd.read_csv(IN / "cholera_model_weekly.csv")
    w["district_raw"] = w["district"]
    w["district"] = w["district"].map(norm_name)
    w["culture_pos"] = w["culture_pos_rate"].fillna(0) * w["culture_tested"]
    sums = ["cases", "deaths", "rf_water_count", "rf_sanitation_count", "rf_hygiene_count",
            "rf_other_count", "rdt_tested", "culture_tested", "culture_pos"]
    reg = (w.dropna(subset=["region"]).assign(region=lambda d: d.region.map(norm_name))
             .groupby("district").region.agg(lambda s: s.mode().iloc[0]))
    p = (w.groupby(["district", "year", "week"], as_index=False)[sums].sum()
           .merge(w.groupby(["district", "year", "week"], as_index=False)["week_start"].first(),
                  on=["district", "year", "week"]))
    p["region"] = p["district"].map(reg)
    p = label_outbreaks_zheng(p)
    p = p.sort_values(["district", "year", "week"]).reset_index(drop=True)
    p.to_csv(OUT / "panel_weekly.csv", index=False)

    old = w.groupby("district_raw").size().size
    print(f"districts: {old} raw IDs -> {p.district.nunique()} after name fix")
    print(f"rows {len(p)} | outbreak weeks {int(p.outbreak.sum())} ({p.outbreak.mean():.4f})")
    print("outbreak weeks per year:", p.groupby("year").outbreak.sum().to_dict())
    print(f"line-list records: {int(p.cases.sum())} | culture-tested {int(p.culture_tested.sum())}"
          f" | culture-positive ~{int(round(p.culture_pos.sum()))}")


if __name__ == "__main__":
    main()
