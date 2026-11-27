#!/usr/bin/env bash
# Runs the full pipeline (about 25 minutes on 2 CPUs).
# Inputs (not included, see README): in/ next to scripts/, or the folder one level up.
# CHIRPS files are read from $CHIRPS_DIR if set, otherwise from <inputs>/chirps/.
set -euo pipefail
cd "$(dirname "$0")"
python3 p01_build_panel.py
python3 p02_era5_district.py
python3 p03_features.py
python3 p04_evaluate.py
python3 p05_figures_and_checks.py
python3 p06_onset_analysis.py
python3 p07_main_table.py
python3 p08_cost_loss_and_yearly_risk.py
python3 p09_leakage_sensitivity.py
python3 p10_rwi.py   # skipped if cmr_relative_wealth_index.csv is missing
python3 p11_wash_lagged.py
