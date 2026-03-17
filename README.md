# UTAR-TE Project

This project reproduces the TE-side experiments for the UTAR paper:
- temporal shift robust anomaly detection on Tennessee Eastman Process (TEP)
- uncertainty-aware gray-zone routing
- selective LLM vs. full LLM cost/performance comparison
- publication-quality figures and tables for the paper

## 1. Directory structure

project_root/
├─ data/
│  ├─ raw/
│  │  ├─ TEP_FaultFree_Training.csv
│  │  ├─ TEP_FaultFree_Testing.csv
│  │  ├─ TEP_Faulty_Training.csv
│  │  └─ TEP_Faulty_Testing.csv
│  ├─ processed/
│  └─ meta/
├─ configs/
├─ src/
│  ├─ data/
│  ├─ models/
│  ├─ routing/
│  ├─ eval/
│  └─ utils/
├─ outputs/
│  ├─ models/
│  ├─ predictions/
│  ├─ metrics/
│  └─ figures/
└─ README.md

## 2. Main experiment concept

### Detection scope
This code focuses on TE detection-side experiments:
- binary detection: normal (0) vs faulty/manipulated (1)
- temporal shift robustness
- UTAR uncertainty routing
- selective LLM call strategy

### Two test versions
We maintain two test sets:

1. full version
- used for main robustness analysis
- PRR, worst-case recall, instability, gray ratio, q-sweep

2. 500 version
- used for selective LLM vs full LLM comparison
- cost analysis
- case study and paper figures

## 3. Required assumptions for TE CSV

The code assumes:
- each CSV contains multiple simulation runs
- there is either an explicit `simulationRun` column or a recoverable run structure
- each run has ordered time steps, typically 500 rows
- faulty samples can be identified by fault/run metadata or by source file
- normal/faulty files follow the common Kaggle-style TE format

The parser is written defensively so that it can adapt to:
- `simulationRun`, `simulation_run`, `run`, `trial`
- `sample`, `time`, `sample_idx`, row-order fallback

## 4. Experiment outputs

### Processed datasets
- data/processed/te_train_rows.csv
- data/processed/te_val_rows.csv
- data/processed/te_test_main_rows.csv
- data/processed/te_test_cost_rows.csv
- data/processed/te_test_full_rows_tcn.csv

### Metadata
- data/meta/feature_columns.json
- data/meta/run_manifest.csv
- data/meta/onset_metadata.csv
- data/meta/split_manifest.json

### Predictions
- outputs/predictions/base_val_predictions.csv
- outputs/predictions/base_test_main_predictions.csv
- outputs/predictions/base_test_cost_predictions.csv
- outputs/predictions/utar_val_selective.csv
- outputs/predictions/utar_test_main_selective.csv
- outputs/predictions/utar_test_main_no_llm.csv
- outputs/predictions/utar_test_cost_selective.csv
- outputs/predictions/utar_test_cost_no_llm.csv
- outputs/predictions/utar_test_cost_full_llm.csv
- outputs/predictions/utar_q_sweep.csv

### Metrics
- outputs/metrics/table1_model_configuration.csv
- outputs/metrics/table2_robustness.csv
- outputs/metrics/table3_q_sweep.csv
- outputs/metrics/table5_ablation.csv
- outputs/metrics/table6_flow_efficiency.csv
- outputs/metrics/table8_grayzone.csv

### Figures
- outputs/figures/figureA_distribution_shift.png
- outputs/figures/figureB_margin_concentration.png
- outputs/figures/figureC_model_discrepancy.png
- outputs/figures/figure2_grayzone_vs_callrate.png
- outputs/figures/figure3_callrate_vs_f1.png
- outputs/figures/figureD_stability_plot.png
- outputs/figures/figure_qsweep_tradeoff.png

## 4.1 Environment

Optional `.env` for OpenAI-backed selective routing:

```bash
OPENAI_API_KEY=sk-pro-...
OPENAI_MODEL=gpt-4o-mini
```

If `configs/routing.yaml` sets `llm.enabled: true` and `llm.mode: openai`, the code loads these values automatically.

## 5. Recommended execution order

### Step 1. Build split
python -m src.data.build_split

This creates:
- row-level train/val/test files
- shift-aware main/cost test files and contiguous TCN test runs
- meta files

# TCN용 시계열 입력을 만드는 단계
python -m src.data.build_windows

# shift 확인 
#test 기준
python -m src.eval.analyze_te_shift \
  --input_csv data/processed/te_test_main_rows.csv \
  --out_dir outputs/shift_analysis_test_main

## val 기준(선택)
python -m src.eval.analyze_te_shift \
  --input_csv data/processed/te_val_rows.csv \
  --out_dir outputs/shift_analysis_val

##
<!-- 입력: te_test_main_rows.csv 같은 row-level split 파일
그룹:
- normal
- transition
- post_shift
- 추가로 shift = transition + post_shift
산출:
feature-wise mean/std difference
KS test p-value / KS statistic
Wasserstein distance
MMD
PCA / t-SNE / UMAP centroid distance
KDE plot
PCA / t-SNE / UMAP 시각화
논문용 요약 CSV -->



### Step 2. Train base models
for s in 0 1 2 3 4; do
  python -m src.models.train_rf --seed $s
  python -m src.models.train_xgb --seed $s
  python -m src.models.train_tcn --seed $s
done


### Step 3. Run base inference
python -m src.models.infer_base_models

### TE test split usage
- `data/processed/te_test_main_rows.csv` is the 4,000-row TE performance test split used for robustness/stability evaluation and selective-LLM performance reporting.
- `data/processed/te_test_cost_rows.csv` is the 500-row TE cost test split used for q-sweep, LLM call-rate, latency, and cost-efficiency analysis.
- In paper terms, Table 2 / Table 5 / Figure D are driven by `test_main`, while Table 3 / Table 6 / Table 8 / Figure 2 / Figure 3 / Figure 5 are driven by `test_cost`.

## threshold + 성능 집계코드 
python -m src.eval.summarize_base_predictions_with_threshold
- outputs/evaluation/best_thresholds_by_seed.json (threshold 저장)
- outputs/evaluation/all_mean_std_table_thresholded.csv (논문표)
- outputs/evaluation/all_seed_metrics_thresholded.csv (seed 별 raw 결과)

### Step 4. Fit threshold and gray-zone
python -m src.routing.fit_thresholds
python -m src.routing.fit_grayzone

### Step 5. Evaluate selective LLM
python -m src.routing.selective_llm_eval_main
- outputs/metrics/main_only_progress
python -m src.routing.selective_llm_eval_cost

- `selective_llm_eval_main`: 4,000-row main set on `q=0.80`, comparing `selective` vs `no_llm`
- `selective_llm_eval_cost`: 500-row cost set on `q=0.60~0.90`, comparing `selective` / `full_llm` / `no_llm`
- `selective_llm_eval`: convenience wrapper that runs both in order


### Step 6. Build paper tables
python -m src.eval.eval_detection
python -m src.eval.eval_shift
python -m src.eval.eval_q_sweep

### Step 7. Generate figures
python -m src.eval.make_figures

## 6. Core paper mapping

### Table 1. Model Configuration Details
Generated by:
- src/eval/eval_detection.py

### Table 2. Robustness and Safety Analysis under Temporal Shift
Generated by:
- src/eval/eval_shift.py

### Table 3. Cost-Performance Pareto Analysis
Generated by:
- src/eval/eval_q_sweep.py

### Table 5. Component-wise Impact on Stability
Generated by:
- src/eval/eval_shift.py
- rows: `Full Framework`, `w/o XGB-Shortcut`, `w/o Gray-Zone`

### Table 6. Data Flow and Filtering Efficiency by Module
Generated by:
- src/routing/selective_llm_eval.py
- src/eval/eval_q_sweep.py

### Table 8. Gray-Zone Performance Comparison
Generated by:
- src/eval/eval_q_sweep.py

### Figure A. Distribution shift
Generated by:
- src/eval/make_figures.py

### Figure B. Margin concentration
Generated by:
- src/eval/make_figures.py

### Figure C. Model discrepancy
Generated by:
- src/eval/make_figures.py

### Figure D. Stability plot
Generated by:
- src/eval/make_figures.py

## 6.1 Command-to-Output Mapping

| Command / Script | Main outputs | Paper mapping |
| --- | --- | --- |
| `python -m src.data.build_split` | `data/processed/te_train_rows.csv`, `data/processed/te_val_rows.csv`, `data/processed/te_test_main_rows.csv`, `data/processed/te_test_cost_rows.csv`, `data/processed/te_test_full_rows_tcn.csv`, `data/meta/run_manifest.csv`, `data/meta/split_manifest.json` | Dataset construction for Sec. 4.2.1 |
| `python -m src.data.build_windows` | `data/processed/te_train_windows.csv`, `data/processed/te_val_windows.csv`, `data/processed/te_test_full_windows_tcn.csv` | TCN input preparation for Sec. 3.2 / Table 1 |
| `python -m src.eval.analyze_te_shift --input_csv data/processed/te_test_main_rows.csv --out_dir outputs/shift_analysis_test_main` | `outputs/shift_analysis_test_main/shift_summary_table.csv`, `feature_shift_normal_vs_shift.csv`, `kde_normal_vs_shift.png`, `pca_phase_scatter.png`, `tsne_phase_scatter.png` | Appendix B, Figure A support |
| `python -m src.models.train_rf --seed s` | `outputs/models/rf_model_seed{s}.pkl`, `outputs/models/rf_imputer_seed{s}.pkl`, `outputs/metrics/rf_val_metrics_seed{s}.json` | Base detector training for Table 1 / Table 2 |
| `python -m src.models.train_xgb --seed s` | `outputs/models/xgb_model_seed{s}.pkl`, `outputs/models/xgb_imputer_seed{s}.pkl`, `outputs/metrics/xgb_val_metrics_seed{s}.json` | Base detector training for Table 1 / Table 2 |
| `python -m src.models.train_tcn --seed s` | `outputs/models/tcn_model_seed{s}.pt`, `outputs/models/tcn_imputer_seed{s}.pkl`, `outputs/models/tcn_scaler_seed{s}.pkl`, `outputs/models/tcn_meta_seed{s}.json`, `outputs/models/tcn_history_seed{s}.csv` | Base detector training for Table 1 / Table 2 |
| `python -m src.models.infer_base_models` | `outputs/predictions/base_val_predictions.csv`, `outputs/predictions/base_test_main_predictions.csv`, `outputs/predictions/base_test_cost_predictions.csv` | Shared input for all TE tables / figures |
| `python -m src.eval.summarize_base_predictions_with_threshold` | `outputs/evaluation/best_thresholds_by_seed.json`, `outputs/evaluation/all_seed_metrics_thresholded.csv`, `outputs/evaluation/all_mean_std_table_thresholded.csv`, split-wise seed summaries | Thresholded baseline summary, Appendix E support |
| `python -m src.routing.fit_thresholds` | `outputs/metrics/thresholds.json` | Eq. 2 threshold calibration, Table 2 / Table 3 / Table 8 support |
| `python -m src.routing.fit_grayzone` | `outputs/metrics/grayzone_grid.csv`, `outputs/metrics/grayzone_defaults.json` | Eq. 2 margin-q grid, Figure 2 / Figure 3 / Table 3 / Table 8 / Appendix A |
| `python -m src.routing.selective_llm_eval_main` | `outputs/predictions/utar_val_selective.csv`, `outputs/predictions/utar_test_main_selective.csv`, `outputs/predictions/utar_test_main_no_llm.csv`, updated `outputs/metrics/selective_llm_summary.csv` | Main 4,000-set validation, `q=0.80`, `selective` vs `no_llm` |
| `python -m src.routing.selective_llm_eval_cost` | `outputs/predictions/utar_test_cost_selective.csv`, `outputs/predictions/utar_test_cost_no_llm.csv`, `outputs/predictions/utar_test_cost_full_llm.csv`, `outputs/predictions/utar_q_sweep.csv`, `outputs/predictions/utar_q_sweep_no_llm.csv`, updated `outputs/metrics/selective_llm_summary.csv` | Cost 500-set q-sweep, `q=0.60~0.90`, `selective` / `full_llm` / `no_llm` |
| `python -m src.routing.selective_llm_eval` | All of the above | Wrapper that runs main first, then cost |
| `python -m src.eval.eval_detection` | `outputs/metrics/table1_model_configuration.csv` | Table 1 |
| `python -m src.eval.eval_shift` | `outputs/metrics/table2_robustness.csv`, `outputs/metrics/table5_ablation.csv` | Table 2, Table 5 |
| `python -m src.eval.eval_q_sweep` | `outputs/metrics/table3_q_sweep.csv`, `outputs/metrics/table6_flow_efficiency.csv`, `outputs/metrics/table8_grayzone.csv` | Table 3, Table 6, Table 8 |
| `MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp python -m src.eval.make_figures` | `outputs/figures/figureA_distribution_shift.png`, `figureB_margin_concentration.png`, `figureC_model_discrepancy.png`, `figure2_grayzone_vs_callrate.png`, `figure3_callrate_vs_f1.png`, `figureD_stability_plot.png`, `figure_qsweep_tradeoff.png` | Figure A, Figure B, Figure C, Figure 2, Figure 3, Figure D, Figure 5 |
| `python -m src.eval.build_appendix_te` | `outputs/appendix/table_a1_q_sweep_base_models.csv`, `table_b1_distribution_shift_summary.csv`, `appendix_b_artifact_manifest.csv`, `table_d1_inference_latency.csv`, `table_e1_seed_variation_detail.csv`, `table_e2_seed_variation_summary.csv` | Appendix A, Appendix B, Appendix D, Appendix E |

## 6.2 Paper Coverage Summary

| Paper item | Output file(s) | Generating script |
| --- | --- | --- |
| Table 1. Model Configuration Details | `outputs/metrics/table1_model_configuration.csv` | `src.eval.eval_detection` |
| Table 2. Robustness and Safety Analysis under Temporal Shift | `outputs/metrics/table2_robustness.csv` | `src.eval.eval_shift` |
| Table 3. Cost-Performance Pareto Analysis | `outputs/metrics/table3_q_sweep.csv` | `src.eval.eval_q_sweep` |
| Table 5. Component-wise Impact on Stability | `outputs/metrics/table5_ablation.csv` | `src.eval.eval_shift` |
| Table 6. Data Flow and Filtering Efficiency by Module | `outputs/metrics/table6_flow_efficiency.csv` | `src.eval.eval_q_sweep` |
| Table 8. Gray-Zone Performance Comparison | `outputs/metrics/table8_grayzone.csv` | `src.eval.eval_q_sweep` |
| Figure A. Distribution shift | `outputs/figures/figureA_distribution_shift.png` | `src.eval.make_figures` |
| Figure B. Margin concentration | `outputs/figures/figureB_margin_concentration.png` | `src.eval.make_figures` |
| Figure C. Model discrepancy | `outputs/figures/figureC_model_discrepancy.png` | `src.eval.make_figures` |
| Figure 2. Gray-Zone Ratio vs LLM Call Rate | `outputs/figures/figure2_grayzone_vs_callrate.png` | `src.eval.make_figures` |
| Figure 3. F1-score vs LLM Call Rate | `outputs/figures/figure3_callrate_vs_f1.png` | `src.eval.make_figures` |
| Figure D. Stability comparison under Temporal Shift | `outputs/figures/figureD_stability_plot.png` | `src.eval.make_figures` |
| Figure 5. Call Rate vs Worst-case Recall trade-off | `outputs/figures/figure_qsweep_tradeoff.png` | `src.eval.make_figures` |
| Appendix A. Gray-zone Ratio and F1-score for individual base models across q-sweep | `outputs/appendix/table_a1_q_sweep_base_models.csv` | `src.eval.build_appendix_te` |
| Appendix B. Distribution Shift Visualization / summary | `outputs/appendix/table_b1_distribution_shift_summary.csv`, `outputs/appendix/appendix_b_artifact_manifest.csv`, `outputs/shift_analysis_test_main/*` | `src.eval.analyze_te_shift`, `src.eval.build_appendix_te` |
| Appendix D. Computational Efficiency: Inference Latency | `outputs/appendix/table_d1_inference_latency.csv` | `src.eval.build_appendix_te` |
| Appendix E. Robustness to Seed Variation | `outputs/appendix/table_e1_seed_variation_detail.csv`, `outputs/appendix/table_e2_seed_variation_summary.csv` | `src.eval.build_appendix_te` |

## 7. Practical notes

### TE evaluation split separation
- `test_main` is the compact 4,000-row TE performance split for selective routing quality evaluation.
- `test_cost` is the compact 500-row TE cost split for LLM usage, latency, and operating-cost analysis.
- The code intentionally separates these two so that performance reporting is not tied to the smaller cost-analysis subset.

### Full vs 500
- full test is the main benchmark
- 500 test is not a substitute for the full benchmark
- 500 is for cost-sensitive routing experiments and visual analysis

### Train balancing
- balance only in train
- do not aggressively rebalance val/test
- preserve temporal structure and post-onset shift regions in test

### Shift definition
The code separates each faulty run into:
- pre
- transition
- post_shift

This is essential for:
- gray-zone analysis
- instability analysis
- worst-case recall
- selective LLM routing benefit

## 8. Limitations

The TE code here reproduces the TE-side experiments.
If NDSS hybrid diagnosis has already been completed separately, keep it as an independent experiment module and only align the reporting format in the final paper.

## 9. Recommended Python environment

pip install numpy pandas scikit-learn matplotlib pyyaml joblib
pip install xgboost
pip install torch

Optional:
pip install umap-learn

## 10. Reproducibility

Use fixed seeds:
- numpy seed
- sklearn random_state
- torch seed

All seeds should be recorded in:
- configs/*.yaml
- outputs/metrics/run_summary.json
