#!/usr/bin/env bash
# Reproduce the submitted output_314551179.csv end to end.
#   bash run_final.sh  PATH_TO_ASSIGNMENT2_REPO
# PATH_TO_ASSIGNMENT2_REPO: the official BigData2026-Assignment2 folder (preprocess.py, test_input.csv)
# RDL_PY: Python of the RDL environment (requirements_rdl.txt), default ../.venv_rdl/Scripts/python
set -euo pipefail
A2=${1:-../BigData2026-Assignment2-main}
RDL_PY=${RDL_PY:-../.venv_rdl/Scripts/python}

# (download.py skips hour files that are already in ../raw_final)
# 1. GH Archive hour files 2026-10-01 00:00 UTC .. 2026-11-20 15:00 UTC (= 11/20 23:59 Taiwan time)
python src/download.py --start 2026-10-01 --end 2026-11-20T16 --out ../raw_final

# 2. the 8 tables, with the official preprocessing script
python "$A2/scripts/preprocess.py" --raw ../raw_final --out ../data_final

# 3. main model: Relational Deep Learning (RelBench) on all 8 tables, trained on weekly
#    snapshots; predicts 11/22 00:00 - 11/29 00:00 (Taiwan time)
$RDL_PY src/rdl.py --mode final --db ../data_final/db --time_enc 1 --epochs 25 --out output_314551179.csv

# 4. supporting model (LightGBM on temporal + relational features), kept for comparison
python src/train.py final --db ../data_final/db --out results/output_lightgbm.csv --tag final_lightgbm

# 5. format check
python "$A2/scripts/evaluate.py" --pred output_314551179.csv --input "$A2/submission/test_input.csv"
