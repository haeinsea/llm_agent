# TE UTAR Dashboard

This dashboard reads the latest UTAR artifacts from the repository root and runs the following pipeline:

- uploaded CSV -> UTAR anomaly scoring
- GraphAD+ Top-K variable extraction and process-graph visualization
- selective LLM routing result display
- sample-level LLM explanation and Q&A

## Components

- `backend/main.py`: FastAPI server and API endpoints
- `backend/models.py`: latest UTAR base models, GraphAD+, and selective routing
- `backend/analysis.py`: GraphAD Top-K extraction, subgraph generation, and structured context for LLM prompts
- `backend/explainer.py`: explanation and Q&A prompts
- `backend/preprocessing.py`: uploaded CSV parsing and feature/label splitting
- `static/index.html`: single-page dashboard UI
- `static/js/app.js`: upload, graph, explanation, and Q&A frontend logic

## UTAR Artifacts Used by the Dashboard

The dashboard does not use any legacy draft models. It reads the latest outputs directly from the repository root.

- base models: `outputs/models/rf_*`, `xgb_*`, `tcn_*`
- GraphAD+: `outputs/models/graphad_artifact.json`
- thresholds: `outputs/metrics/thresholds.json`
- gray-zone grid: `outputs/metrics/grayzone_grid.csv`
- selected q: `outputs/metrics/selected_q.json`
- feature schema: `data/meta/feature_columns.json`

## Uploaded CSV Format

The sample file is `TE_test_sample_for_dashboard.csv`.

- The best results come from including all 41 `xmeas_*` and 11 `xmv_*` columns.
- If a `fault` column exists, the dashboard also shows the label and phase.
- If one of `Unnamed: 0`, `index`, `sample_idx`, `sample`, or `id` exists, it is used as the display index.
- Missing features are filled with `NaN` and automatically aligned to the 52-feature schema used by UTAR.

## Current Logic

### 1. Anomaly Detection

Uploaded samples follow the latest UTAR inference path.

- RF: 10-seed mean probability
- XGBoost: 10-seed mean probability
- ModernTCN: 10-seed mean probability
- UTAR base score: routing-feature-based score derived from the three detectors above
- Selective routing: final decision using the current gray-zone margin and threshold from `selected_q.json`

Key values shown at upload time:

- `UTAR base score`
- `RF / XGBoost / ModernTCN`
- `ensemble mean`
- `ensemble entropy`
- `model discrepancy`
- `tau`
- `selected q`
- `gray-zone flag`
- `LLM called`
- `final decision source`

### 2. GraphAD+ Variable Visualization

GraphAD+ uses the latest `graphad_artifact.json`.

- sensor graph: correlation graph saved during training
- score: smoothed anomaly score from GraphAD+ inference
- top-k: direct use of `graphad_topk_sensors` and `graphad_topk_scores`
- direction: `increase` / `decrease` based on the uploaded sample versus the uploaded-data median
- subgraph: 1-hop process graph around the Top-K variables

This means the dashboard no longer uses the old pseudo-ranking based on `GraphAD(model) × |z-score|`; it uses the current GraphAD+ outputs from the paper/codebase directly.

### 3. LLM Explanations

The LLM is used in two places.

- routing LLM: selective routing for gray-zone samples
- explanation LLM: root-cause explanation and process-path summary for the selected sample

The explanation prompt includes:

- UTAR base / RF / XGBoost / ModernTCN / final score
- selected q / tau / gray-zone / llm_called / decision_source
- GraphAD+ Top-K variables
- process graph context
- subgraph context
- meta features

## How to Run

1. Make sure the required artifacts have already been generated at the repository root.
2. To enable OpenAI-based explanations, set `OPENAI_API_KEY` either in the dashboard folder or at the repository root.
3. Install dependencies and start the server.

```bash
cd te_dashboard_full_graphad
uvicorn backend.main:app --reload
```

4. Open the dashboard in a browser.

- `http://localhost:8000`

## Usage Flow

1. Upload a TEP CSV such as `TE_test_sample_for_dashboard.csv`
2. Select a sample row
3. Inspect the UTAR anomaly score and final decision
4. Review the GraphAD+ Top-K variables and process subgraph
5. Click `Generate Explanation for Current Sample` to request an LLM explanation
6. Use the Q&A box to ask follow-up questions about the selected sample

## Notes

- The dashboard allows a stub fallback for routing LLM calls so that upload analysis still works even if network or OpenAI calls fail.
- Explanation quality depends on whether the OpenAI connection is available.
