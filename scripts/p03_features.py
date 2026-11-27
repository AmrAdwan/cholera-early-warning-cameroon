"""Step 3. Merge panel + climate and build lagged features.

All features for week t use only information from earlier weeks:
  climate lags 1-2 and 'operational' lags 2-3 (climate data arrive a few days late,
  so at the start of week t the last complete available week is t-2);
  case history (autoregressive benchmark): cases in weeks t-1, t-2 and t-1..t-4,
  and a delayed version (t-2, t-3, t-2..t-5) assuming one week of reporting delay;
  WASH risk-factor counts lagged one week. In the same week these counts are only
  non-zero when cases occur, so they would leak the outcome.
The outbreak label itself is not lagged into features: labels at t-1 depend on
cases at t and t+1, so they would leak the future.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, pandas as pd

CLIM = ["mean_temp", "mean_dewpoint", "total_precip", "max_precip"]  # district climate (CHIRPS rainfall)
RF = ["rf_water_count", "rf_sanitation_count", "rf_hygiene_count", "rf_other_count"]


def add_lags(df, cols, lags, suffix=""):
    g = df.groupby("district")
    for c in cols:
        for L in lags:
            df[f"{c}{suffix}_lag{L}"] = g[c].shift(L)
    return df


def main():
    panel = pd.read_csv(OUT / "panel_weekly.csv")
    dist = pd.read_csv(OUT / "era5_weekly_district.csv")
    # climate on the full 2017-2024 weekly grid so 2018 lags exist
    grid = dist.merge(panel, on=["district", "year", "week"], how="left")
    grid = grid.sort_values(["district", "year", "week"]).reset_index(drop=True)
    grid = add_lags(grid, CLIM, [1, 2, 3])
    grid["log_cases"] = np.log1p(grid["cases"].fillna(0))
    g = grid.groupby("district")["log_cases"]
    grid["cases_lag1"], grid["cases_lag2"], grid["cases_lag3"] = g.shift(1), g.shift(2), g.shift(3)
    for L in (1, 2):  # log of cases summed over the 4 weeks ending L weeks before t
        grid[f"cases_4wk_lag{L}"] = np.log1p(grid.groupby("district")["cases"].transform(
            lambda s, L=L: s.fillna(0).shift(L).rolling(4, min_periods=1).sum()))
    grid = add_lags(grid, RF, [1])
    for c in RF:
        grid[c] = grid[c].fillna(0); grid[c + "_lag1"] = grid[c + "_lag1"].fillna(0)
    df = grid[grid.year.between(2018, 2024) & grid.outbreak.notna()].copy()
    df["outbreak"] = df["outbreak"].astype(int)
    df.to_csv(OUT / "features_weekly.csv", index=False)
    print(df.shape, "| NaN in key lags:", int(df[["total_precip_lag3", "cases_lag2"]].isna().sum().sum()))


if __name__ == "__main__":
    main()
