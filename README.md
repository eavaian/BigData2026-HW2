# Predicting GitHub User Activity (Big Data Assignment 2)

For each of the 4,465 test users, predict whether the user will **open a PR, open an issue,
comment on an issue/PR, review a PR, comment on PR code, or push** during
2026/11/22 00:00 – 11/29 00:00 (Taiwan time), from the 8 relational tables built by the
official `preprocess.py`.

**Main model: Relational Deep Learning (RDL, Lecture 5 / RelBench).** The 8 tables become one
heterogeneous graph (one node per row, one edge type per foreign key). A graph neural network
with a temporal encoder learns from it directly, without hand-made features. It is trained on
weekly snapshots, with a threshold for each activity. `output_314551179.csv` is its prediction.

**Supporting model: LightGBM on hand-made temporal + relational features.** It is used to
compare against, and to show which relations and time signals carry the information.

## Data Source

The data is from **[GH Archive](https://www.gharchive.org/)**
(https://www.gharchive.org/), a public archive of GitHub events. It is used only for this
course assignment. All tables are produced and anonymized by the course's
`scripts/preprocess.py`.

**Period used:** GH Archive hour files **2026-10-01 00:00 UTC – 2026-11-20 15:00 UTC** (1,216 hour
files, none missing) = data up to **11/20 23:59 Taiwan time**. The same data is used for all
experiments and for `output_314551179.csv`:

| Use | Weeks (Taiwan time) |
|---|---|
| validation (3 weeks, all experiments) | 11/14 – 11/21 (newest), 11/07 – 11/14, 10/31 – 11/07 |
| final prediction | trained on label weeks up to 11/21 00:00; predicts 11/22 – 11/29 |

## Setup

Two Python environments are used:

```bash
# 1. data, LightGBM, analyses (tested with Python 3.14 on Windows)
pip install -r requirements.txt

# 2. RDL, the main model (Python 3.12; torch, PyG, PyTorch Frame, relbench)
py -3.12 -m venv ../.venv_rdl
../.venv_rdl/Scripts/python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
../.venv_rdl/Scripts/python -m pip install -r requirements_rdl.txt
```

`run_final.sh` and `run_dev.sh` find the RDL Python at `../.venv_rdl/Scripts/python`; set
`RDL_PY=...` to use another one (e.g. `../.venv_rdl/bin/python` on Linux / macOS).

Windows notes: Smart App Control blocked several compiled libraries on our machine, so
- `pyarrow==22.0.0` is pinned, because pyarrow 26 could not load its parquet DLL;
- `pyg_lib` is not installed, because its DLL was blocked. RDL therefore does not use PyG's
  neighbour sampler (see Method, step 4);
- scikit-learn could not load its compiled modules. The comparison models in `study.py` are
  written in numpy or use LightGBM.

## How to run

The official assignment repository (with `scripts/preprocess.py`, `scripts/evaluate.py` and
`submission/test_input.csv`) is expected at `../BigData2026-Assignment2-main`. Pass another
path as the first argument if it is somewhere else.

```bash
# reproduce the submission (download -> preprocess -> RDL -> predict -> format check)
bash run_final.sh ../BigData2026-Assignment2-main

# reproduce the validation experiments, figures and tables of the report
bash run_dev.sh ../BigData2026-Assignment2-main
```

What `run_final.sh` runs, step by step:

```bash
python src/download.py --start 2026-10-01 --end 2026-11-20T16 --out ../raw_final
python ../BigData2026-Assignment2-main/scripts/preprocess.py --raw ../raw_final --out ../data_final
../.venv_rdl/Scripts/python src/rdl.py --mode final --db ../data_final/db --time_enc 1 --epochs 25 \
       --out output_314551179.csv                                     # main model, ~10-15 min on CPU
python src/train.py final --db ../data_final/db --out results/output_lightgbm.csv --tag final_lightgbm
python ../BigData2026-Assignment2-main/scripts/evaluate.py --pred output_314551179.csv \
       --input ../BigData2026-Assignment2-main/submission/test_input.csv
```

Randomness is fixed (`torch.manual_seed(2026)`; LightGBM `random_state=2026`,
`deterministic=True`). RDL results can still change slightly with the number of CPU threads,
because the order of floating-point sums changes. `run_final.sh` uses 8 threads (the default);
the validation runs in `run_dev.sh` use 3 threads each so that two can run in parallel.

## Code

| File | What it does |
|---|---|
| `src/download.py` | Downloads GH Archive hour files for a UTC range. Resumable; uses a plain User-Agent because the default Python one gets HTTP 403. |
| `src/config.py` | All time settings (UTC): data period, validation week, gap, weekly snapshots. |
| `src/data.py` | Loads the 8 tables and builds one activity log (user, activity, time, row time, repo), used for the labels. Enforces the row-time rule. |
| **`src/rdl.py`** | **Main model.** RelBench database and graph, temporal subgraphs, GNN training, thresholds; `--mode dev` validates on 3 weeks, `--mode final` writes the submission. Options for the ablations: `--time_enc`, `--layers`, `--drop_tables`. |
| `src/features.py` | Supporting model: temporal (`T_*`), cohort-wide (`G_*`) and relational (`R_*`) features at a cutoff. |
| `src/train.py` | Supporting model: 6 LightGBM classifiers on the features, thresholds, validation or prediction. Also the copy baselines. |
| `src/metrics.py` | Same score as `evaluate.py`: (macro-F1 + micro-F1) / 2. |
| `src/results_io.py` | Writes `results/experiments.csv` under a lock file (runs can finish at the same time). |
| `src/experiments.py` | Runs one LightGBM configuration on 3 validation weeks and appends it to `results/experiments.csv`. |
| `src/gnn.py` | Smaller comparison GNN on the user–repository graph only (6 relation types). Plain PyTorch. |
| `src/study.py` | Other models on the same features, finer ablations, thresholds, training weeks, user segments, paired bootstrap (also for the saved RDL predictions). |
| `src/analysis.py` | Figures (data gaps, positive rates, feature importance, results, RDL ablation) and the random-vs-time split analysis. |
| `src/schema_graph.py` | Q1 schema graph (`results/figs/schema_graph.png`, `results/schema.dot`). |

## Method (main model, `src/rdl.py`)

This is the RelBench pipeline from Lecture 5:

1. **Database.** The 8 tables become a relbench `Database` with their primary keys, foreign
   keys and row-time columns (`first_seen_at` / `created_at`). The columns that are known only
   later (`merged_at`, `merged_by_id`, `closed_at`, `state_reason`, `dismissed_at`) are
   dropped: the graph filters rows by row time, and it cannot hide single column values.
   Coded ids are dropped, masked text is replaced by its length, and other time columns become
   their age in hours relative to the row time.
2. **Graph.** `make_pkey_fkey_graph` builds a heterogeneous graph: one node per row (8 node
   types: Users 6,069, Repositories 12,928, PullRequests 15,271, Issues 9,733, IssueComments
   3,820, PRReviews 3,787, PRReviewComments 1,834, Pushes 43,637; 97,079 rows in total). Each
   foreign key is an edge type in both directions. That is 19 foreign keys and 38 edge types,
   because `merged_by_id` was dropped in step 1. PyTorch Frame's `HeteroEncoder` encodes the
   columns of each table.
3. **Training table.** Each weekly cutoff *c* gives one row per cohort user who has at least one
   activity before *c*, with 6 labels: did the user do the activity in `[c + gap, c + gap + 7 days)`.
   The validation gap is 0. The final gap is 1 day, because the data stops 1 day before the test
   week. Training uses up to 5 weeks before the predicted week. A snapshot needs at least 5 days
   of history, and the data starts on 10/01, so the validation weeks starting 11/14, 11/07 and
   10/31 have 5, 4 and 3 training weeks. The final run has 5 (label weeks starting 10/17 – 11/14).
4. **Temporal subgraphs.** For each cutoff *c*, only rows whose row time is before *c* (and the
   edges between them) are kept, so no row from the future is reachable. This is temporal
   neighbour sampling with all neighbours. PyG's sampler needs `pyg_lib`, which Windows
   application control blocked on our machine. With ~100k rows, full-batch message passing on
   these time-filtered graphs is cheap.
5. **Model.** `HeteroEncoder` → `HeteroTemporalEncoder` → 2-layer `HeteroGraphSAGE` (64 channels,
   sum aggregation) → MLP head with 6 outputs. The temporal encoder encodes the time between each
   row and the cutoff and adds it to the row's embedding.
6. **Training.** Binary cross-entropy for the 6 outputs. The positive weight is
   sqrt(negatives / positives). Adam with lr 0.005 and mini-batches of 2,048 seed users of one
   snapshot. The model trains on all weeks except the newest for up to 25 epochs. The newest week
   picks the epoch (early stopping) and, for each activity, the probability threshold with the
   best F1. The early-stopped model is kept (no refit, see Results).
7. **Prediction.** Test users with no activity before the cutoff are predicted 0.
8. **Push.** In GH Archive, push events are mostly missing after Oct 27: the cohort has
   ~2,000 – 3,000 pushes per day before, 0 from 10/27 to 11/02 and 0 – 300 per day after. The
   weekly push positive rate falls from 56% (week of 10/10) to 4 – 10% (weeks of 10/31 – 11/14),
   following data coverage, not user behaviour. Push is still
   predicted by the model with a threshold tuned on the newest week. If GH Archive is missing push
   data in the test week, push is not scored.

## Supporting model (LightGBM, `src/train.py`)

Same snapshots, labels and threshold rule. Instead of the graph, each user gets ~116 hand-made
features at the cutoff:
- **Temporal (82):** counts in 1/3/7/14/28-day windows; exponentially decayed counts
  (half-life 2/7/21 days); days since the last activity of each type and since the first;
  active days and weeks; week-over-week trend; weekend and night share.
- **Relational (34):** repositories touched and how concentrated the activity is; owned and
  organisation repositories; how busy those repositories are; collaborators (users sharing a
  repository) and how active they are; the user's open PRs and issues; reviews and comments the
  user received; PRs merged by the user; review relationships; the open-PR backlog in
  repositories the user reviews.

The known-later columns are used only when their value is before the cutoff. There is one
LightGBM per activity (7 leaves, 150 trees; chosen in an earlier round of development). Thresholds
are tuned on the newest week, then the model is refit on all weeks.

## Results (validation, mean of 3 weeks: 11/14, 11/07, 10/31 Taiwan time)

All numbers are x100 and come from `results/experiments.csv`, `results/study_*.csv`,
`results/leakage.json` and `results/figs/`. `results/rdl_pred/` holds the RDL predictions for
each test user and week. "w/o push" leaves out push, because push labels mainly reflect GH Archive
gaps. The 95% CIs are paired bootstraps over the 4,465 test users (1,000 resamples per week,
`study.py`).

| Method | Score (6 activities) | Score w/o push |
|---|---|---|
| Copy last week (official baseline) | 32.82 | 35.18 |
| Copy last 14 days | 32.17 | 35.77 |
| GNN on the user–repo graph only + time encoding (`gnn.py`)* | 34.70 | 39.78 |
| **RDL on 8 tables + time encoding (main)** | **36.98** | **39.90** |
| LightGBM, temporal + relational features (supporting) | 37.66 | 40.13 |

\* `gnn.py` copies the last 14 days for push, so compare it on the w/o-push column.

Per validation week (score / w/o push):

| Week | RDL (main) | LightGBM |
|---|---|---|
| 11/14 | 39.82 / 40.71 | 37.88 / 41.86 |
| 11/07 | 35.69 / 38.89 | 37.36 / 39.15 |
| 10/31 | 35.42 / 40.10 | 37.74 / 39.37 |

- RDL and LightGBM are statistically tied. LightGBM − RDL = +0.68, 95% CI [-0.38, +1.77] on the
  full score and +0.23 [-0.96, +1.37] without push. RDL is better in the newest week (11/14).
- RDL is +4.16 [+2.83, +5.50] above copying last week, without any hand-made features.
- Without push, RDL on all 8 tables is about equal to the GNN on the user–repo graph only
  (39.90 vs 39.78); on the full score it is clearly better (36.98 vs 34.70), because it predicts push
  with the model instead of copying it.

### RDL ablation (`results/figs/rdl_ablation.png`, `results/rdl_ablation.csv`)

| RDL variant | Score | Score w/o push | Diff vs main, full score [95% CI] | Diff vs main, w/o push [95% CI] |
|---|---|---|---|---|
| **main: 8 tables, 2 layers, time encoding** | **36.98** | **39.90** | — | — |
| no time encoding | 36.75 | 39.69 | -0.23 [-0.87, +0.42] | -0.21 [-0.98, +0.57] |
| 1 layer (1 hop) | 36.29 | 39.44 | -0.69 [-1.29, -0.01] | -0.46 [-1.18, +0.33] |
| without Pushes | 36.29 | 40.03 | -0.69 [-1.52, +0.09] | +0.13 [-0.83, +1.02] |
| without PRReviews + PRReviewComments | 34.09 | 36.39 | -2.88 [-3.67, -2.02] | -3.51 [-4.50, -2.45] |
| without Issues + IssueComments | 33.59 | 34.66 | -3.39 [-4.32, -2.48] | -5.24 [-6.38, -4.13] |
| refit on all weeks after tuning thresholds | 35.00 | 37.58 | | |

- **Tables:** removing the review tables or the issue tables costs 3.5 – 5.2 points w/o push. The
  loss is mostly in the activity that the removed table records: without Issues + IssueComments,
  open_issue F1 falls from 0.42 – 0.47 to 0.26 – 0.29 across the 3 weeks; without the review tables,
  pr_review F1 falls from 0.36 – 0.44 to 0.29 – 0.30. The model relies most on the user's own past
  rows of that activity.
- **Second hop:** a second GraphSAGE layer adds +0.69 on the full score (CI just above 0) and +0.46
  w/o push (not significant).
- **Pushes table:** helps only the push column (full score -0.69 without it, w/o push +0.13).
- **Time encoding:** better in all 3 weeks w/o push (40.71 vs 40.55, 38.89 vs 38.57, 40.10 vs 39.93),
  but the gain is small (+0.21) and not significant; 70% of the bootstrap samples favour it.
- **Refit:** refitting on all weeks after tuning the thresholds hurts RDL (35.00 / 37.58). The
  thresholds no longer match the refitted network's probabilities, so RDL keeps the early-stopped
  model. (For LightGBM the refit also costs a little on these weeks: 38.47 / 40.59 without refit vs
  37.66 / 40.13 with it; the refit setting was chosen in an earlier round of development and kept.)

### Error analysis by user segment (`results/figs/study_segments.png`, `results/study_segments.csv`)

Users are grouped by days since their last activity before the cutoff (F1 pooled over activities
and the 3 weeks):

| Segment | Share of positives | Copy last week | LightGBM | RDL |
|---|---|---|---|---|
| active < 1 day ago | 22% | 0.52 | 0.58 | 0.56 |
| 1–7 days ago | 40% | 0.34 | 0.42 | 0.42 |
| 7–28 days ago | 32% | 0 | 0.12 | 0.17 |
| > 28 days ago | 3% | 0 | 0 | 0.05 |
| no history (predicted 0) | 2% | 0 | 0 | 0 |

RDL finds more of the users who return after a break: for 7–28 days, recall 0.21 vs 0.08 for
LightGBM, at lower precision (0.14 vs 0.22). F1 per activity (pooled over the 3 weeks,
`results/study_errors.csv`):

| Activity | RDL | LightGBM | Copy last week |
|---|---|---|---|
| open_pr | 0.41 | 0.43 | 0.35 |
| open_issue | 0.45 | 0.43 | 0.41 |
| issue_comment | 0.41 | 0.41 | 0.36 |
| pr_review | 0.39 | 0.41 | 0.36 |
| pr_review_comment | 0.30 | 0.28 | 0.25 |
| push | 0.26 | 0.27 | 0.24 |

### Supporting-model analyses (LightGBM, `study.py`, `experiments.py`)

These show which relations and time signals carry information, with the same 3 weeks.

**Other models on the same features** (w/o push; diff vs LightGBM with 95% CI): copy last 7 days
35.18; decayed count of the activity with a tuned threshold, no learning, 39.03 (-1.09 [-2.22, +0.06]);
logistic regression 39.39 (-0.74 [-1.60, +0.23]); MLP 36.29 (-3.84 [-5.02, -2.68]); random forest
40.55 (+0.42 [-0.34, +1.24]); LightGBM with balanced class weights 40.69 (+0.57 [-0.14, +1.32]);
LightGBM with 31 leaves x 500 trees 37.37 (-2.75 [-3.75, -1.74], overfits). Logistic regression and
the MLP are numpy implementations; the random forest is LightGBM in random-forest mode.

**Feature-group ablation** (`results/figs/study_ablation.png`; full model 40.13 w/o push):

| Change | Score w/o push |
|---|---|
| drop PR / issue state (open PRs, merged PRs, open / assigned issues) | 38.53 |
| drop all temporal features (R only) | 38.68 |
| drop all relational features (T only) | 38.72 |
| drop interactions (reviews / comments received, review relationships) | 38.99 |
| drop window counts | 39.31 |
| drop decayed counts | 39.35 |
| only the last 7 days, no relational features | 39.38 |
| temporal features of the predicted activity only (no cross-activity signal) | 39.67 |
| drop collaborators | 39.97 |
| drop repository activity | 40.03 |
| drop is_bot | 40.13 |
| drop own repositories | 40.46 |

- The relational features add +1.40 over temporal features only, 95% CI [+0.63, +2.17].
- Adding collaborator-share features (share of the activity done by users who share a repository)
  lowers the score to 39.56, so they are not used.
- The most important feature for open_pr (week 11/14 model) is the number of PRs the user merged for
  others in the last 28 days (21% of the gain).

**Time handling:**
- Thresholds tuned on the newest week: 40.13. Top-k (k = positives in the newest week): 40.09. A
  fixed 0.5 threshold: 27.80.
- Training weeks: 2 → 38.52, 3 → 40.20, 4 → 39.95, 5 → 40.13.
- A 1-day gap between the data and the label week (the final setting): 38.46 (vs 40.13 with no gap).
- Splitting the same rows **at random** instead of by time gives 47.59 instead of 37.88 on the
  newest week (`results/leakage.json`), mostly through push (F1 0.63 vs 0.16): a random split leaks
  the future and is far too optimistic.

## Final run (11/21)

`bash run_final.sh` on the data up to 11/20 23:59 Taiwan time: RDL stopped at epoch 20 of 25 (mean
F1 0.424 on the newest week); thresholds 0.29 (push) to 0.58 (pr_review_comment). Predicted users
out of 4,465, with the actual count of the newest observed week (11/14 – 11/21) for comparison:

| | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push |
|---|---|---|---|---|---|---|
| predicted, 11/22 – 11/29 | 515 | 185 | 261 | 316 | 91 | 528 |
| actual, 11/14 – 11/21 | 332 | 206 | 204 | 227 | 78 | 402 |

## Output

`output_314551179.csv`: one row per user in `test_input.csv`, with the header
`UserID,open_pr,open_issue,issue_comment,pr_review,pr_review_comment,push` and 0/1 values,
predicted by the RDL model.
