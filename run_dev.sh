#!/usr/bin/env bash
# Reproduce the validation experiments, figures and tables of the report.
#   bash run_dev.sh  PATH_TO_ASSIGNMENT2_REPO
set -euo pipefail
A2=${1:-../BigData2026-Assignment2-main}

# data: 2026-10-01 00:00 UTC .. 2026-11-20 15:59 UTC (= 11/20 23:59 Taiwan time), the same as
# run_final.sh (skip these two lines if run_final.sh has already built ../data_final).
# Validation weeks (Taiwan time): 11/14 - 11/21, 11/07 - 11/14, 10/31 - 11/07.
python src/download.py --start 2026-10-01 --end 2026-11-20T16 --out ../raw_final
python "$A2/scripts/preprocess.py" --raw ../raw_final --out ../data_final
D=../data_final/db

# baselines and the final configuration, each on 3 validation weeks
python src/experiments.py --db $D --name base_copy7 -- --baseline copy7
python src/experiments.py --db $D --name base_copy14 -- --baseline copy14
python src/experiments.py --db $D --name final_config
python src/experiments.py --db $D --name push_copy14 -- --push_rule copy14
# feature-group ablations (Q2 / Q3)
python src/experiments.py --db $D --name abl_T -- --groups T
python src/experiments.py --db $D --name abl_TG -- --groups T G
python src/experiments.py --db $D --name abl_RG_noT -- --groups R G
python src/experiments.py --db $D --name abl_TRG -- --groups T R G
python src/experiments.py --db $D --name nb_TR -- --drop
python src/experiments.py --db $D --name q3_no_decay -- --drop R_nb_ R_repo_share_ T_decay
python src/experiments.py --db $D --name q3_no_since -- --drop R_nb_ R_repo_share_ T_days_since T_mean_gap
python src/experiments.py --db $D --name q3_no_profile -- --drop R_nb_ R_repo_share_ T_weekend T_night T_trend
python src/experiments.py --db $D --name q3_gap1 -- --gap_days 1
for w in 2 3; do python src/experiments.py --db $D --name weeks$w -- --weeks $w; done
python src/experiments.py --db $D --name abl_R_only -- --groups R
python src/experiments.py --db $D --name rate_threshold -- --thr_mode rate
# GNN comparison
python src/gnn.py --db $D --time_enc 1
python src/gnn.py --db $D --time_enc 0
# Relational Deep Learning on all 8 tables (RelBench; separate Python 3.12 env, requirements_rdl.txt)
RDL_PY=${RDL_PY:-../.venv_rdl/Scripts/python}
# main model, then its ablations (each saves per-user predictions in results/rdl_pred/)
$RDL_PY src/rdl.py --db $D --time_enc 1 --epochs 25 --threads 3
$RDL_PY src/rdl.py --db $D --time_enc 0 --epochs 25 --threads 3
$RDL_PY src/rdl.py --db $D --layers 1 --name rdl_layers1 --threads 3
$RDL_PY src/rdl.py --db $D --drop_tables Pushes --name rdl_no_pushes --threads 3
$RDL_PY src/rdl.py --db $D --drop_tables PRReviews PRReviewComments --name rdl_no_reviews --threads 3
$RDL_PY src/rdl.py --db $D --drop_tables Issues IssueComments --name rdl_no_issues --threads 3
$RDL_PY src/rdl.py --db $D --time_enc 1 --epochs 15 --refit 1 --threads 3
$RDL_PY src/rdl.py --db $D --time_enc 0 --epochs 15 --refit 1 --threads 3
python src/experiments.py --db $D --name lgbm_norefit -- --refit 0
# figures, leakage analysis, schema graph
# model comparison, finer ablations, thresholds, training weeks, segments, bootstrap CIs
python src/study.py --db $D
python src/analysis.py --db $D
python src/schema_graph.py
