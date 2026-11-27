# Cholera early warning in Cameroon: an operational evaluation of machine learning

Code for the paper *How much does climate add? An operational evaluation of machine learning
for cholera early warning in Cameroon* (in preparation).

The pipeline predicts, one week ahead, whether a health district in Cameroon is in a cholera
outbreak week, and evaluates the predictions the way they would be used for anticipatory
action:

- expanding-window rolling-origin evaluation over the test years 2020 to 2024;
- decision thresholds chosen on out-of-fold predictions only;
- benchmarks based on recent case counts (persistence and case history);
- operational scores: probability of detection (POD), false alarm ratio (FAR), false alarms in an
  outbreak-free year, skill for the first week of an outbreak, and relative economic value over
  cost/loss ratios;
- robustness checks: nested tuning, outbreak thresholds from training years only, reporting
  delay, data latency, Relative Wealth Index, lagged WASH risk-factor counts and districts
  without earlier outbreaks.

## Data

**The cholera surveillance data are not included.** They were obtained from the Cameroon
Ministry of Public Health under a non-disclosure agreement involving the Cameroon Red Cross and
the Netherlands Red Cross, and cannot be shared. `.gitignore` excludes all data and results.

The scripts expect these inputs in a folder `in/` next to `scripts/` (or in the folder one level
above the repository):

| File | Content | Source |
|---|---|---|
| `cholera_model_weekly.csv` | weekly cases and WASH risk-factor counts per health district | national cholera line list (not shared) |
| `data_stream-oper_stepType-instant.nc` | ERA5 2 m temperature and dewpoint, 0.25 deg | Copernicus Climate Data Store |
| `data_stream-oper_stepType-accum.nc` | ERA5 accumulated fields (read by step 2; rainfall is taken from CHIRPS) | Copernicus Climate Data Store |
| `chirps/chirps-v2.0.YYYY.days_p25.nc` | CHIRPS v2.0 daily rainfall, 0.25 deg, 2017 to 2024 | Climate Hazards Center, UC Santa Barbara |
| `cmr_admin1/2/3.*` | administrative boundaries of Cameroon | Humanitarian Data Exchange |
| `cmr_relative_wealth_index.csv` | Relative Wealth Index (Chi et al. 2022) | Humanitarian Data Exchange |

CHIRPS files can also be read from another folder by setting `CHIRPS_DIR`.

## Running

```bash
pip install -r requirements.txt
bash scripts/run_all.sh        # about 25 minutes on 2 CPUs
```

Results (tables as CSV, figures as PNG) are written to `results/`.

| Step | Script | What it does |
|---|---|---|
| 1 | `p01_build_panel.py` | district-week panel, merged district names, outbreak labels (Zheng et al. 2022) |
| 2 | `p02_era5_district.py` | weekly district climate: CHIRPS rainfall, ERA5 temperature and dewpoint |
| 3 | `p03_features.py` | lagged climate, case history and WASH features |
| 4 | `p04_evaluate.py` | rolling-origin evaluation, threshold rules, scores |
| 5 | `p05_figures_and_checks.py` | variance inflation factors, figures, permutation importance |
| 6 | `p06_onset_analysis.py` | skill for the first week of an outbreak |
| 7 | `p07_main_table.py` | compact results table |
| 8 | `p08_cost_loss_and_yearly_risk.py` | relative economic value and yearly district ranking |
| 9 | `p09_leakage_sensitivity.py` | nested tuning; outbreak thresholds from training years only |
| 10 | `p10_rwi.py` | Relative Wealth Index added |
| 11 | `p11_wash_lagged.py` | lagged WASH risk-factor counts added |

## Notes

- No health-district boundary file was available, so districts are matched by name to communes
  or departments (`results/district_geometry_match.csv`). This should be replaced by
  health-district boundaries when available.
- The ERA5 input files contain only the 00 UTC time step, so temperature and dewpoint are
  night-time values.

## Contact

Amr Adwan, Vrije Universiteit Amsterdam
