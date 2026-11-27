"""Step 2. District-level weekly climate.

No health-district shapefile is available, so each health district is matched by
name to a commune (ADM3) or department (ADM2). Urban health districts in Douala,
Yaounde, Garoua etc. get the union of their city's communes; at 0.25 deg (~28 km)
the climate grids cannot separate them anyway. The match table is saved for checking
and should be replaced by real health-district boundaries when available.

Rainfall: CHIRPS v2.0 daily at 0.25 deg (files chirps-v2.0.YYYY.days_p25.nc from
data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/netcdf/p25/, read from
$CHIRPS_DIR or <inputs>/chirps/). Weekly total and maximum daily rainfall.
Temperature and dewpoint: ERA5 at 0.25 deg; the input files hold the 00 UTC step only.
District values are the mean of the grid cells whose centres fall inside the polygon
(nearest cell if none); weeks are ISO weeks.
"""
import sys, os; sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from common import *
import numpy as np, pandas as pd, geopandas as gpd, xarray as xr
from shapely.geometry import Point
from shapely.ops import unary_union

CITY = {  # health district -> communes (ADM3) whose union is used
    **{k: [f"DOUALA {i}" for i in ["1ER", "2E", "3E", "4E", "5E", "6E"]] for k in
       ["BANGUE", "BOKO", "BONASSAMA", "CITE DES PALMIERS", "DEIDO", "JAPOMA", "LOGBABA", "NEW BELL", "NYLON", "MANOKA"]},
    **{k: [f"YAOUNDE {i}" for i in ["1ER", "2E", "3E", "4E", "5E", "6E", "7E"]] for k in
       ["BIYEM ASSI", "CITE VERTE", "DJOUNGOLO", "MVOG ADA", "NKOLBISSON", "NKOLNDONGO", "ODZA"]},
    "BERTOUA": ["BERTOUA 1ER", "BERTOUA 2E"], "EBOLOWA": ["EBOLOWA 1ER", "EBOLOWA 2E"], "EDEA": ["EDEA 1ER", "EDEA 2E"],
    "GAROUA I": ["GAROUA 1ER", "GAROUA 2E", "GAROUA 3E"], "GAROUA II": ["GAROUA 1ER", "GAROUA 2E", "GAROUA 3E"],
    "KRIBI": ["KRIBI 1ER", "KRIBI 2E"], "KUMBA NORTH": ["KUMBA 1ST", "KUMBA 2ND", "KUMBA 3RD"],
    "KUMBA SOUTH": ["KUMBA 1ST", "KUMBA 2ND", "KUMBA 3RD"], "LIMBE": ["LIMBE 1ST", "LIMBE 2ND", "LIMBE 3RD"],
    "NKONGSAMBA": ["NKONGSAMBA 1ER", "NKONGSAMBA 2E", "NKONGSAMBA 3E"], "BAKASSI": ["IDABATO", "KOMBO ABEDIMO"],
}
DEPT = {"ABO": "MOUNGO", "GASCHIGA": "BENOUE", "GOLOMBE": "BENOUE", "NGONG": "BENOUE", "MADA": "LOGONE ET CHARI"}


def district_geoms(names):
    a3 = gpd.read_file(IN / "cmr_admin3.shp").to_crs(4326); a3["k"] = a3.adm3_name1.map(norm_name)
    a2 = gpd.read_file(IN / "cmr_admin2.shp").to_crs(4326); a2["k"] = a2.adm2_name1.map(norm_name)
    g3, g2 = a3.groupby("k").geometry.agg(unary_union), a2.groupby("k").geometry.agg(unary_union)
    rows = []
    for d in names:
        if d in CITY:
            rows.append((d, unary_union([g3[c] for c in CITY[d]]), "city communes (ADM3 union)"))
        elif d in g3.index:
            rows.append((d, g3[d], "ADM3 name match"))
        elif d in g2.index:
            rows.append((d, g2[d], "ADM2 name match"))
        elif d in DEPT:
            rows.append((d, g2[DEPT[d]], f"department {DEPT[d]} (approximate)"))
        else:
            raise ValueError(f"no geometry for {d}")
    return gpd.GeoDataFrame(rows, columns=["district", "geometry", "match"], crs=4326)


def main():
    panel = pd.read_csv(OUT / "panel_weekly.csv")
    gdf = district_geoms(sorted(panel.district.unique()))
    daily_dir = IN / "era5_daily"
    if daily_dir.exists() and list(daily_dir.glob("era5_daily_*.nc")):
        inst = acc = xr.open_mfdataset(sorted(daily_dir.glob("era5_daily_*.nc")), combine="by_coords").load()
        print("ERA5 source: era5_daily/ (all 24 hours)")
    else:
        inst = xr.open_dataset(IN / "data_stream-oper_stepType-instant.nc")
        acc = xr.open_dataset(IN / "data_stream-oper_stepType-accum.nc")
        print("ERA5 source: input files (00 UTC only)")
    lat, lon = inst.latitude.values, inst.longitude.values
    LON, LAT = np.meshgrid(lon, lat)
    pts = gpd.GeoSeries([Point(x, y) for x, y in zip(LON.ravel(), LAT.ravel())], crs=4326)

    cells, ncell = {}, []
    for d, geom in zip(gdf.district, gdf.geometry):
        inside = np.where(pts.within(geom).values)[0]
        if len(inside) == 0:  # small polygon: nearest cell to centroid
            c = geom.centroid; inside = np.array([np.argmin((LON.ravel() - c.x) ** 2 + (LAT.ravel() - c.y) ** 2)])
        cells[d] = inside; ncell.append(len(inside))
    gdf["n_era5_cells"] = ncell
    gdf.drop(columns="geometry").to_csv(OUT / "district_geometry_match.csv", index=False)

    # Cameroon-wide series (land mask), saved for reference
    cmr = unary_union(gpd.read_file(IN / "cmr_admin1.shp").to_crs(4326).geometry)
    cells["__CAMEROON__"] = np.where(pts.within(cmr).values)[0]

    var = {"temp": inst.t2m.values - 273.15, "dewpoint": inst.d2m.values - 273.15,
           "precip": acc.tp.values * 1000, "runoff": acc.sro.values * 1000}
    T = var["temp"].shape[0]
    flat = {k: v.reshape(T, -1) for k, v in var.items()}
    t = pd.to_datetime(inst.valid_time.values)
    iso = t.isocalendar()
    out = []
    for d, idx in cells.items():
        dd = pd.DataFrame({k: np.nanmean(v[:, idx], axis=1) for k, v in flat.items()})
        dd["year"], dd["week"] = iso.year.values, iso.week.values
        wk = dd.groupby(["year", "week"]).agg(mean_temp=("temp", "mean"), mean_dewpoint=("dewpoint", "mean"),
                                               total_precip=("precip", "sum"), total_runoff=("runoff", "sum")).reset_index()
        wk["district"] = d; out.append(wk)
    w = pd.concat(out, ignore_index=True)

    # CHIRPS v2.0 daily rainfall (0.25 deg) replaces the one-hour ERA5 precipitation and runoff.
    # Files chirps-v2.0.YYYY.days_p25.nc from data.chc.ucsb.edu/products/CHIRPS/v2.0/global_daily/netcdf/p25/
    # are read from $CHIRPS_DIR or <inputs>/chirps/. Weekly: total rainfall and maximum daily rainfall.
    chirps_dir = Path(os.environ.get("CHIRPS_DIR", IN / "chirps"))
    files = sorted(chirps_dir.glob("chirps-v2.0.*.days_p25.nc"))
    if files:
        ch = xr.concat([xr.open_dataset(f).sel(latitude=slice(1.0, 14.0), longitude=slice(7.5, 17.0)).load() for f in files], dim="time")
        clat, clon = ch.latitude.values, ch.longitude.values
        CLON, CLAT = np.meshgrid(clon, clat)
        cpts = gpd.GeoSeries([Point(x, y) for x, y in zip(CLON.ravel(), CLAT.ravel())], crs=4326)
        geoms = dict(zip(gdf.district, gdf.geometry)); geoms["__CAMEROON__"] = cmr
        pr = ch.precip.values.reshape(len(ch.time), -1)
        ct = pd.to_datetime(ch.time.values); ciso = ct.isocalendar()
        rows = []
        for d, geom in geoms.items():
            inside = np.where(cpts.within(geom).values)[0]
            if len(inside) == 0:
                c = geom.centroid; inside = np.array([np.argmin((CLON.ravel() - c.x) ** 2 + (CLAT.ravel() - c.y) ** 2)])
            dd = pd.DataFrame({"precip": np.nanmean(pr[:, inside], axis=1), "year": ciso.year.values, "week": ciso.week.values})
            wk = dd.groupby(["year", "week"]).precip.agg(total_precip="sum", max_precip="max").reset_index()
            wk["district"] = d; rows.append(wk)
        cw = pd.concat(rows, ignore_index=True)
        w = w.drop(columns=["total_precip", "total_runoff"]).merge(cw, on=["district", "year", "week"], how="left")
        print(f"Rainfall source: CHIRPS daily ({len(files)} files, {ct.min().date()} to {ct.max().date()})")
    else:
        print("Rainfall source: ERA5 files (CHIRPS not found)")
    w[w.district == "__CAMEROON__"].drop(columns="district").to_csv(OUT / "era5_weekly_cameroon_masked.csv", index=False)
    w = w[w.district != "__CAMEROON__"]
    w.to_csv(OUT / "era5_weekly_district.csv", index=False)
    print(gdf.match.value_counts().to_string()); print("cells per district: median", int(np.median(ncell)), "min", min(ncell))
    sd = w.groupby(["year", "week"]).total_precip.std().mean()
    print(f"weekly rows {len(w)} | mean between-district SD of weekly precip: {sd:.3f} (was 0 with national data)")


if __name__ == "__main__":
    main()
