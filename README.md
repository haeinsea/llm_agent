# UTAR-TE Project

This repository reproduces the TE-side experiments for the updated UTAR manuscript.

Current scope:
- TE anomaly detection under temporal distribution shift
- UTAR gray-zone routing with configurable seeds (see `configs/experiment.yaml`)
- `ModernTCN` as the temporal backbone
- GraphAD+ structural context for selective LLM prompting
- paper-ready tables, figures, and appendices for the TE manuscript

Out of scope here:
- NDSS-specific diagnosis experiments
- the manual framework illustration in `Figure 1`
- the dashboard mockup / UI concept figure

## 1. Dataset and split policy

The project keeps the current `build_split.py` structure.

- `train`: model fitting split
- `val`: threshold, routing, and hyperparameter validation split
- `test_main`: 4,000 rows for robustness / stability reporting
- `test_cost`: 400 rows for LLM cost / latency / operating-policy reporting
- `test_full`: contiguous temporal test set used to align `ModernTCN` and GraphAD+ inference back to `test_main` and `test_cost`

This means:
- robustness tables and stability figures come from `test_main`
- q-sweep selection comes from `test_main`
- final cost / latency / operating-policy comparison comes from `test_cost`
- temporal backbone and GraphAD+ are measured on contiguous full test runs, then allocated back to the compact reporting splits

## 2. Manuscript alignment

The code is aligned to the updated manuscript with these decisions.

- Seed policy is controlled in `configs/experiment.yaml`, and manuscript wording should match that file.
- The temporal base model is `ModernTCN`, not legacy `TCN`.
- UTAR weighting follows the manuscript logic:
  - per-run temporal normalization
  - sigmoid temporal weight transition
  - final UTAR base score from RF / XGB / temporal competition
- `w/o Graph Smoothing` now uses raw GraphAD context instead of smoothed GraphAD context, which is closer to the manuscript wording than simply removing GraphAD entirely.
- base robustness reporting follows the configured seed list, and LLM-invoked routing modes follow the per-split policy in `configs/experiment.yaml`.
- Appendix H hyperparameter rationale is driven by current configs and, if available, by the tuning outputs in `outputs/tuning/*_best.json`.

Current seed-policy source of truth:
- `configs/experiment.yaml` sets the active seed list to `0..9`.
- base detectors (`RF`, `XGB`, `ModernTCN`) follow the full configured seed list.
- SOTA temporal baselines (`AdapTable`-style, invariant) follow the full configured seed list.
- UTAR main / q-sweep evaluation follows `main_llm_seed_policy`.
- UTAR cost evaluation follows `cost_llm_seed_policy`, which is currently `representative`.
- existing files under `outputs/` may still contain old `0..4` results until you rerun the pipeline.

`<참고사항>`-driven experimental intent reflected in code:
- `w/o Selective LLM`: gray-zone samples stay on UTAR base routing
- `w/o Gray-Zone (Routing)`: simple ensemble comparison
- `w/o Graph Smoothing`: raw GraphAD context
- `w/o Ensemble Entropy`: all gray-zone samples escalate to LLM
- `4.7.1` / `4.7.2`: performance-drop and prediction-volatility figures are generated

## 3. Generated paper artifacts

Auto-generated TE artifacts now cover:

- methodology evidence figure above the framework figure
- Figure A / B / C
- Figure 2 / 3 / 4 / 5 / 6
- the extra stability plot used to support Eq. (5)
- Table 1 / 2 / 3 / 5 / 6 / 7 / 8 / 9 / 10
- Appendix B / C / D / E / F / G / H

Manual or excluded items:
- framework illustration `Figure 1`
- dashboard mockup figure
- NDSS diagnosis figures / tables outside the Appendix H hyperparameter evidence bundle

## 4. Key outputs

Main generated files:

- `outputs/metrics/table1_model_configuration.csv`
- `outputs/metrics/table2_robustness.csv`
- `outputs/metrics/table3_q_sweep.csv`
- `outputs/metrics/table5_ablation.csv`
- `outputs/metrics/table6_flow_efficiency.csv`
- `outputs/metrics/table7_entropy_filter_effect.csv`
- `outputs/metrics/table8_grayzone.csv`
- `outputs/metrics/table9_dashboard_modules.csv`
- `outputs/metrics/table10_operational_strategy.csv`

- `outputs/figures/figure_methodology_evidence.png`
- `outputs/figures/figureA_distribution_shift.png`
- `outputs/figures/figureB_margin_concentration.png`
- `outputs/figures/figureC_model_discrepancy.png`
- `outputs/figures/figureD_stability_plot.png`
- `outputs/figures/figure2_qsweep_elbow.png`
- `outputs/figures/figure3_callrate_vs_f1.png`
- `outputs/figures/figure4_performance_drop.png`
- `outputs/figures/figure5_prediction_flips.png`
- `outputs/figures/figure6_cost_stability_pareto.png`

- `outputs/appendix/table_b1_prompt_structure.csv`
- `outputs/appendix/table_b2_prompt_variables.csv`
- `outputs/appendix/table_a0_baseline_descriptions.csv`
- `outputs/appendix/table_a1_supplementary_experimental_results.csv`
- `outputs/appendix/table_c1_failure_case_summary.csv`
- `outputs/appendix/table_c2_failure_case_examples.csv`
- `outputs/appendix/figure_c1_failure_case_profiles.png`
- `outputs/appendix/table_d1_q_sweep_base_models.csv`
- `outputs/appendix/table_d2_inference_latency.csv`
- `outputs/appendix/table_e1_distribution_shift_summary.csv`
- `outputs/appendix/table_f1_graphad_rank_trace.csv`
- `outputs/appendix/figure_f1_graphad_visual_proof.png`
- `outputs/appendix/table_g1_seed_variation_detail.csv`
- `outputs/appendix/table_g2_seed_variation_summary.csv`
- `outputs/appendix/table_h1_hyperparameter_rationale.csv`
- `outputs/appendix/table_h2_graphad_lambda_selection.csv`
- `outputs/appendix/table_h2_graphad_lambda_grid_detail.csv`
- `outputs/appendix/table_h2_graphad_lambda_selection_ndss.csv`
- `outputs/appendix/table_h2_graphad_lambda_grid_detail_ndss.csv`
- `outputs/appendix/figure_h2_graphad_lambda_sensitivity_ndss.png`
- `outputs/appendix/table_h6_graphad_alpha_selection_ndss.csv`
- `outputs/appendix/table_h6_graphad_alpha_grid_detail_ndss.csv`
- `outputs/appendix/table_h7_graphad_alpha_protocol_ndss.csv`
- `outputs/appendix/table_h8_graphad_alpha_reproducibility_ndss.csv`
- `outputs/appendix/table_h9_graphad_alpha_current_vs_best_ndss.csv`
- `outputs/appendix/figure_h3_graphad_alpha_sensitivity_ndss.png`

## 5. Recommended execution order

If you want a clean rerun from the latest code, use this order.

### 5.1 Clean stale generated outputs

```bash
python -m src.eval.clean_generated_outputs
```

### 5.2 Build data splits and shift analysis

```bash
python -m src.data.build_split
python -m src.data.build_windows

python -m src.eval.analyze_te_shift \
  --input_csv data/processed/te_test_main_rows.csv \
  --out_dir outputs/shift_analysis_test_main
```

Optional:

```bash
python -m src.eval.analyze_te_shift \
  --input_csv data/processed/te_val_rows.csv \
  --out_dir outputs/shift_analysis_val
```

### 5.3 Optional base-model and GraphAD optimization

These scripts search candidate settings and write the best results to `outputs/tuning`.

```bash
python -m src.tuning.optimize_rf
python -m src.tuning.optimize_xgb
python -m src.tuning.optimize_modern_tcn
python -m src.tuning.optimize_graphad
```

Notes:
- `optimize_rf` / `optimize_xgb` search static base-model parameters on `val`
- `optimize_modern_tcn` searches `ModernTCN` architecture and training knobs
- `optimize_graphad` searches `corr_threshold`, `alpha`, and GraphAD weights using TE-side proxy objectives:
  - anomaly separation
  - candidate consistency across shifted runs
  - perturbation robustness

After search, review:

- `outputs/tuning/rf_best.json`
- `outputs/tuning/xgb_best.json`
- `outputs/tuning/tcn_best.json`
- `outputs/tuning/graphad_best.json`

Then update the corresponding `configs/*.yaml` before the final retraining run.

If you only retune `ModernTCN`, the minimum rerun path is:

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m src.models.train_tcn --seed $s
done

python -m src.models.infer_base_models
python -m src.eval.summarize_base_predictions_with_threshold
python -m src.models.infer_sota_models
python -m src.eval.eval_sota_baselines
```

### 5.4 Train base models

```bash
for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m src.models.train_rf --seed $s
  python -m src.models.train_xgb --seed $s
  python -m src.models.train_tcn --seed $s
done

python -m src.models.train_graphad
```

### 5.5 Run base inference

```bash
python -m src.models.infer_base_models
python -m src.eval.summarize_base_predictions_with_threshold
```

This stage produces:
- base detector predictions for `val`, `test_main`, and `test_cost`
- GraphAD+ columns for prompt context
- runtime summaries for RF / XGB / ModernTCN / GraphAD+ / UTAR base stack

### 5.5a Train SOTA baselines for Appendix A0 / A1

```bash
python -m src.models.train_sota_models
python -m src.models.infer_sota_models
python -m src.eval.eval_sota_baselines
```

This stage produces:
- 10-seed `AdapTable`-style and invariant temporal baseline artifacts
- `outputs/predictions/sota_*.csv`
- `outputs/appendix/table_a0_baseline_descriptions.csv`
- `outputs/appendix/table_a1_supplementary_experimental_results.csv`

### 5.5b End-to-end 10-seed execution summary

Use this sequence when you want every model-side artifact refreshed under the current 10-seed policy.

```bash
python -m src.eval.clean_generated_outputs
python -m src.data.build_split
python -m src.data.build_windows

for s in 0 1 2 3 4 5 6 7 8 9; do
  python -m src.models.train_rf --seed $s
  python -m src.models.train_xgb --seed $s
  python -m src.models.train_tcn --seed $s
done

python -m src.models.train_graphad
python -m src.models.infer_base_models
python -m src.eval.summarize_base_predictions_with_threshold

python -m src.models.train_sota_models
python -m src.models.infer_sota_models
python -m src.eval.eval_sota_baselines

python -m src.routing.fit_thresholds
python -m src.routing.fit_grayzone
python -m src.routing.selective_llm_eval_main
python -m src.eval.eval_q_sweep
python -m src.routing.selective_llm_eval_cost

python -m src.eval.eval_detection
python -m src.eval.eval_shift
python -m src.eval.eval_system_tables
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp python -m src.eval.make_figures
python -m src.eval.build_appendix_te
python -m src.eval.update_reference_docx
```

### 5.6 Optional routing optimization

`optimize_routing` depends on `outputs/predictions/base_val_predictions.csv`, so it must be run after base inference.

```bash
python -m src.tuning.optimize_routing
```

Notes:
- `optimize_routing` searches `q`, sigmoid gain, and shortcut quantiles
- it uses stub LLM logic only
- after review, update `configs/routing.yaml` if you want to lock in the selected setting

### 5.7 Fit UTAR threshold and gray-zone

```bash
python -m src.routing.fit_thresholds
python -m src.routing.fit_grayzone
```

### 5.8 Run selective LLM evaluation

```bash
python -m src.routing.selective_llm_eval_main
python -m src.eval.eval_q_sweep
python -m src.routing.selective_llm_eval_cost
```

This writes:
- `utar_val_*`
- `utar_test_main_*`
- `utar_q_sweep.csv`
- `utar_q_sweep_no_llm.csv`
- `selected_q.json`
- `utar_test_cost_*`
- merged seed metrics and summary tables

q-sweep efficiency:
- `selective_llm_eval_main` warms the largest q in `grayzone_grid.csv` first, so lower-q selective runs reuse cached LLM responses instead of repeating shared API calls.
- `eval_q_sweep` only aggregates saved `utar_q_sweep*.csv` outputs and does not make any LLM calls.
- OpenAI prompt/response cache is also written to `outputs/metrics/selective_llm_response_cache.jsonl`, so restarting the same routing experiment can reuse prior calls unless you clear outputs.

Policy:
- base detector and non-LLM routing baselines follow the configured seed list
- LLM-invoked routing modes follow the per-split seed policy in `configs/experiment.yaml`
- q selection is performed on the 4,000-sample `test_main` routing set
- the selected q is then fixed for the 400-sample `test_cost` final comparison

If you only need to refresh the selected-q ablation rows for `Table 5`, you do not need to rerun the full q-sweep. Use:

```bash
python -m src.routing.selective_llm_eval_main --selected-q-only
python -m src.eval.eval_shift
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp python -m src.eval.build_appendix_te
python -m src.eval.update_reference_docx
```

This shortcut updates the selected `q` on `val/main` for:
- `selective`
- `selective_no_graph`
- `selective_no_filter`
- `no_llm`
- `ensemble_only`

If the full-framework row is already finalized and you only want the remaining ablations at the selected `q` while preserving older default-q files, use:

```bash
python -m src.routing.selective_llm_eval_main --selected-q-only --modes no_llm ensemble_only selective_no_graph selective_no_filter
python -m src.eval.eval_shift
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp python -m src.eval.build_appendix_te
python -m src.eval.update_reference_docx
```

In this mode, prediction files are saved with a q tag such as `utar_test_main_selective_no_graph_q060.csv`, so existing `q=0.80` files are not overwritten.
If you only want to append extra q-sweep rows such as `q=0.20` and `q=0.40` for `Table 3` / `Table 8` while keeping the current selected `q`, use:

```bash
python -m src.routing.fit_grayzone --q-values 0.20 0.40
python -m src.routing.selective_llm_eval_main --q-values 0.20 0.40
python -m src.eval.eval_q_sweep --keep-selected-q
python -m src.eval.update_reference_docx
```

This shortcut only adds the requested q rows for the main `selective` / `no_llm` q-sweep outputs and preserves the selected-q interpretation already used in the manuscript.
- Appendix B prompt example prefers the representative cost-routing output

### 5.9 Build paper tables

```bash
python -m src.eval.eval_detection
python -m src.eval.eval_shift
python -m src.eval.eval_q_sweep
python -m src.eval.eval_system_tables
```

### 5.10 Generate figures and appendices

```bash
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp python -m src.eval.make_figures
python -m src.eval.build_appendix_te
```

## 6. Script-to-output mapping

| Command | Main outputs | Manuscript use |
| --- | --- | --- |
| `python -m src.eval.clean_generated_outputs` | cleaned `outputs/metrics`, `outputs/figures`, `outputs/appendix`, `outputs/evaluation`, selected `outputs/predictions`, selected `outputs/tuning` | clean rerun |
| `python -m src.data.build_split` | row-level train / val / test files | split construction |
| `python -m src.data.build_windows` | `te_train_windows.csv`, `te_val_windows.csv`, `te_test_full_windows_tcn.csv` | `ModernTCN` input |
| `python -m src.eval.analyze_te_shift ...` | shift summary CSVs and KDE / PCA / t-SNE visuals | Appendix E, Figure A support |
| `python -m src.tuning.optimize_rf` | `outputs/tuning/rf_trials.csv`, `outputs/tuning/rf_best.json` | Appendix H support |
| `python -m src.tuning.optimize_xgb` | `outputs/tuning/xgb_trials.csv`, `outputs/tuning/xgb_best.json` | Appendix H support |
| `python -m src.tuning.optimize_modern_tcn` | `outputs/tuning/tcn_trials.csv`, `outputs/tuning/tcn_best.json` | Appendix H support |
| `python -m src.tuning.optimize_routing` | `outputs/tuning/routing_trials.csv`, `outputs/tuning/routing_best.json` | Appendix H support |
| `python -m src.tuning.optimize_graphad` | `outputs/tuning/graphad_trials.csv`, `outputs/tuning/graphad_best.json` | Appendix H support |
| `python -m src.models.train_rf --seed s` | RF models and validation metrics | base detector training |
| `python -m src.models.train_xgb --seed s` | XGB models and validation metrics | base detector training |
| `python -m src.models.train_tcn --seed s` | `ModernTCN` models and histories | base detector training |
| `python -m src.models.train_graphad` | `graphad_artifact.json`, GraphAD summary | GraphAD+ fit |
| `python -m src.models.train_sota_models` | `adaptable_tcn_*`, `invariant_tcn_*` model families across the configured seed list | Appendix A0 / A1 baseline training |
| `python -m src.models.infer_base_models` | `base_*.csv`, base runtime summaries | shared input for TE figures / tables |
| `python -m src.models.infer_sota_models` | `sota_val_predictions.csv`, `sota_test_main_predictions.csv`, `sota_test_cost_predictions.csv` | Appendix A1 baseline inference |
| `python -m src.eval.summarize_base_predictions_with_threshold` | thresholded baseline summaries | Table 2 / Appendix G support |
| `python -m src.eval.eval_sota_baselines` | `table_a0_baseline_descriptions.csv`, `table_a1_supplementary_experimental_results.csv`, `sota_seed_metrics.csv` | Appendix A0 / A1 |
| `python -m src.routing.fit_thresholds` | `thresholds.json`, `thresholds_by_seed.csv` | UTAR threshold calibration |
| `python -m src.routing.fit_grayzone` | `grayzone_grid.csv`, `grayzone_grid_by_seed.csv`, `grayzone_defaults.json` | q-sweep and margin analysis |
| `python -m src.routing.selective_llm_eval_main` | `utar_val_*`, `utar_test_main_*`, `utar_q_sweep*.csv`, merged summary update | main-side routing evaluation and q candidates |
| `python -m src.eval.eval_q_sweep` | `table3_q_sweep.csv`, `table6_flow_efficiency.csv`, `table8_grayzone.csv`, `selected_q.json` | q selection on the 4,000-sample routing set |
| `python -m src.routing.selective_llm_eval_cost` | `utar_test_cost_*`, merged summary update | final 400-sample comparison with selected q |
| `python -m src.eval.eval_detection` | `table1_model_configuration.csv` | Table 1 |
| `python -m src.eval.eval_shift` | `table2_robustness.csv`, `table5_ablation.csv` | Table 2, Table 5 |
| `python -m src.eval.eval_system_tables` | `table7_entropy_filter_effect.csv`, `table9_dashboard_modules.csv`, `table10_operational_strategy.csv` | Table 7, Table 9, Table 10 |
| `python -m src.eval.make_figures` | paper figures under `outputs/figures` | methodology evidence, Figure A/B/C/2/3/4/5/6, stability plot |
| `python -m src.eval.build_appendix_te` | appendix tables / figures under `outputs/appendix` | Appendix B/C/D/E/F/G/H |

## 7. Paper coverage summary

| Paper item | Output file | Script |
| --- | --- | --- |
| methodology evidence figure above Figure 1 | `outputs/figures/figure_methodology_evidence.png` | `src.eval.make_figures` |
| Figure A | `outputs/figures/figureA_distribution_shift.png` | `src.eval.make_figures` |
| Figure B | `outputs/figures/figureB_margin_concentration.png` | `src.eval.make_figures` |
| Figure C | `outputs/figures/figureC_model_discrepancy.png` | `src.eval.make_figures` |
| Figure 4 | `outputs/figures/figure4_performance_drop.png` | `src.eval.make_figures` |
| Figure 5 | `outputs/figures/figure5_prediction_flips.png` | `src.eval.make_figures` |
| Figure 6 | `outputs/figures/figure6_cost_stability_pareto.png` | `src.eval.make_figures` |
| stability support figure for Eq. (5) | `outputs/figures/figureD_stability_plot.png` | `src.eval.make_figures` |
| Table 1 | `outputs/metrics/table1_model_configuration.csv` | `src.eval.eval_detection` |
| Table 2 | `outputs/metrics/table2_robustness.csv` | `src.eval.eval_shift` |
| Table 3 | `outputs/metrics/table3_q_sweep.csv` | `src.eval.eval_q_sweep` |
| Table 5 | `outputs/metrics/table5_ablation.csv` | `src.eval.eval_shift` |
| Table 6 | `outputs/metrics/table6_flow_efficiency.csv` | `src.eval.eval_q_sweep` |
| Table 7 | `outputs/metrics/table7_entropy_filter_effect.csv` | `src.eval.eval_system_tables` |
| Table 8 | `outputs/metrics/table8_grayzone.csv` | `src.eval.eval_q_sweep` |
| Table 9 | `outputs/metrics/table9_dashboard_modules.csv` | `src.eval.eval_system_tables` |
| Table 10 | `outputs/metrics/table10_operational_strategy.csv` | `src.eval.eval_system_tables` |
| Appendix A0 | `outputs/appendix/table_a0_baseline_descriptions.csv` | `src.eval.eval_sota_baselines` |
| Appendix A1 | `outputs/appendix/table_a1_supplementary_experimental_results.csv`, `outputs/appendix/table_a1_supplementary_seed_detail.csv` | `src.eval.eval_sota_baselines` |
| Appendix B | `outputs/appendix/table_b1_prompt_structure.csv`, `outputs/appendix/table_b2_prompt_variables.csv`, `outputs/appendix/appendix_b_prompt_example.txt` | `src.eval.build_appendix_te` |
| Appendix C | `outputs/appendix/table_c1_failure_case_summary.csv`, `outputs/appendix/table_c2_failure_case_examples.csv`, `outputs/appendix/figure_c1_failure_case_profiles.png` | `src.eval.build_appendix_te` |
| Appendix D | `outputs/appendix/table_d1_q_sweep_base_models.csv`, `outputs/appendix/table_d2_inference_latency.csv` | `src.eval.build_appendix_te` |
| Appendix E | `outputs/appendix/table_e1_distribution_shift_summary.csv`, `outputs/appendix/appendix_e_artifact_manifest.csv` | `src.eval.build_appendix_te` |
| Appendix F | `outputs/appendix/table_f1_graphad_rank_trace.csv`, `outputs/appendix/figure_f1_graphad_visual_proof.png` | `src.eval.build_appendix_te` |
| Appendix G | `outputs/appendix/table_g1_seed_variation_detail.csv`, `outputs/appendix/table_g2_seed_variation_summary.csv` | `src.eval.build_appendix_te` |
| Appendix H | `outputs/appendix/table_h1_hyperparameter_rationale.csv`, `outputs/appendix/table_h2_graphad_lambda_selection.csv`, `outputs/appendix/table_h2_graphad_lambda_grid_detail.csv`, `outputs/appendix/table_h2_graphad_lambda_selection_ndss.csv`, `outputs/appendix/table_h2_graphad_lambda_grid_detail_ndss.csv`, `outputs/appendix/table_h3_graphad_lambda_protocol_ndss.csv`, `outputs/appendix/table_h4_graphad_lambda_reproducibility_ndss.csv`, `outputs/appendix/figure_h2_graphad_lambda_sensitivity_ndss.png` | `src.eval.build_appendix_te`, `ndss.tune_graphad_weights` |

## 8. Environment

Recommended packages:

```bash
pip install numpy pandas scikit-learn matplotlib pyyaml joblib
pip install xgboost
pip install torch
```

Optional:

```bash
pip install umap-learn
```

Optional `.env` for OpenAI-backed selective routing:

```bash
OPENAI_API_KEY=sk-pro-...
OPENAI_MODEL=gpt-4o-mini
```

Recommended Apple Silicon training environment:

```bash
conda activate utar-mps
export KMP_DUPLICATE_LIB_OK=TRUE
python -c "import torch; print(torch.backends.mps.is_available())"
```

Expected output on Apple Silicon:
- `True`

If you train from this environment, the temporal PyTorch models should select `mps` automatically.

## 9. Reproducibility notes

- seed list is defined in `configs/experiment.yaml`
- if `configs/experiment.yaml` is missing, code defaults to 10 seeds
- current repository outputs may still reflect older 5-seed runs; verify regenerated CSVs before using them in the manuscript
- manuscript-side seed count should match the code
- all final paper outputs should be regenerated after tuning or config changes
- if outputs predate a code change, clean first and rerun the full sequence above

##q실험
python -m src.routing.fit_grayzone --q-values 0.20 0.40
python -m src.routing.selective_llm_eval_main --q-values 0.20 0.40
python -m src.eval.eval_q_sweep --keep-selected-q
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp python -m src.eval.build_appendix_te
python -m src.eval.update_reference_docx


## table 5 확정 시 
python -m src.routing.selective_llm_eval_main --selected-q-only --modes ensemble_only selective_no_graph selective_no_filter
python -m src.eval.eval_shift
MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp python -m src.eval.build_appendix_te
python -m src.eval.update_reference_docx
