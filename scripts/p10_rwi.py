"""Step 10. Socio-economic predictor: Relative Wealth Index (Chi et al. 2022).

Needs the Meta Relative Wealth Index file for Cameroon from HDX
(search "Relative Wealth Index Cameroon" on data.humdata.org; CSV with columns
latitude, longitude, rwi, error). Save it as `cmr_relative_wealth_index.csv` in the
inputs folder. If the file is missing, this step prints a note and stops.

RWI is static, so each district gets one value: the mean RWI of the ~2.4 km tiles inside
its geometry (same geometry matching as p02; nearest tiles if none fall inside).
Then the main configurations are rerun with RWI added.
"""
import sys; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import p04_evaluate as E
from p02_era5_district import district_geoms
import numpy as np, pandas as pd, geopandas as gpd

RWI_FILE = IN / "cmr_relative_wealth_index.csv"


def district_rwi(names):
    r = pd.read_csv(RWI_FILE)
    pts = gpd.GeoDataFrame(r, geometry=gpd.points_from_xy(r.longitude, r.latitude), crs=4326)
    geoms = district_geoms(names)
    j = gpd.sjoin(pts, geoms[["district", "geometry"]], predicate="within")
    out = j.groupby("district").rwi.agg(["mean", "size"]).rename(columns={"mean": "rwi", "size": "n_tiles"})
    for d in set(names) - set(out.index):  # tiny polygon without tiles: 5 nearest tiles to the centroid
        c = geoms.set_index("district").loc[d, "geometry"].centroid
        near = r.assign(d2=(r.longitude - c.x) ** 2 + (r.latitude - c.y) ** 2).nsmallest(5, "d2")
        out.loc[d] = [near.rwi.mean(), 0]
    return out.reset_index()


def main():
    if not RWI_FILE.exists():
        print(f"RWI file not found ({RWI_FILE.name}); download it from HDX first. Skipping step 10.")
        return
    df = pd.read_csv(OUT / "features_weekly.csv")
    rwi = district_rwi(sorted(df.district.unique())); rwi.to_csv(OUT / "10_district_rwi.csv", index=False)
    df = df.merge(rwi[["district", "rwi"]], on="district", how="left")
    E.FEATURE_SETS["Climate district + RWI"] = E.CLIM_D + ["rwi"]
    E.FEATURE_SETS["Climate district + case history + RWI"] = E.CLIM_D + E.AR + ["rwi"]
    cfg = [(fs, m) for fs in ["Climate district", "Climate district + RWI", "Climate district + case history",
                              "Climate district + case history + RWI"] for m in ["RF", "LR"]]
    res, _ = E.run(df, cfg, [2020, 2021, 2022, 2023, 2024], tag="RWI")
    res.to_csv(OUT / "10_rwi_by_year.csv", index=False)
    s = res[(res.test_year <= 2023) & (res.threshold_rule == "POD>=0.8")].groupby(["features", "model"])[["AUC", "POD", "FAR"]].mean()
    s.round(3).to_csv(OUT / "10_rwi_summary.csv"); print(s.round(3).to_string())


if __name__ == "__main__":
    main()
