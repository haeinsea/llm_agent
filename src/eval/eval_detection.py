from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.temporal_backbone import temporal_model_display_name
from src.utils.io import read_yaml, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"


def main() -> None:
    rf_cfg = read_yaml(CONFIG_DIR / "train_rf.yaml", default={})
    xgb_cfg = read_yaml(CONFIG_DIR / "train_xgb.yaml", default={})
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    graphad_cfg = read_yaml(CONFIG_DIR / "train_graphad.yaml", default={})
    routing_cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})
    temporal_name = temporal_model_display_name(tcn_cfg.get("architecture", "modern_tcn"))

    rows = [
        {"Component": "RF", "Parameter": "n_estimators", "Value": rf_cfg.get("n_estimators")},
        {"Component": "RF", "Parameter": "max_depth", "Value": rf_cfg.get("max_depth")},
        {"Component": "RF", "Parameter": "min_samples_split", "Value": rf_cfg.get("min_samples_split")},
        {"Component": "RF", "Parameter": "min_samples_leaf", "Value": rf_cfg.get("min_samples_leaf")},
        {"Component": "XGB", "Parameter": "n_estimators", "Value": xgb_cfg.get("n_estimators")},
        {"Component": "XGB", "Parameter": "max_depth", "Value": xgb_cfg.get("max_depth")},
        {"Component": "XGB", "Parameter": "learning_rate", "Value": xgb_cfg.get("learning_rate")},
        {"Component": "XGB", "Parameter": "subsample", "Value": xgb_cfg.get("subsample")},
        {"Component": "XGB", "Parameter": "colsample_bytree", "Value": xgb_cfg.get("colsample_bytree")},
        {"Component": "XGB", "Parameter": "reg_lambda", "Value": xgb_cfg.get("reg_lambda")},
        {"Component": temporal_name, "Parameter": "architecture", "Value": tcn_cfg.get("architecture", "modern_tcn")},
        {"Component": temporal_name, "Parameter": "window_size", "Value": tcn_cfg.get("window_size")},
        {"Component": temporal_name, "Parameter": "stride", "Value": tcn_cfg.get("stride")},
        {"Component": temporal_name, "Parameter": "channels", "Value": ",".join(map(str, tcn_cfg.get("channels", [])))},
        {"Component": temporal_name, "Parameter": "dilations", "Value": ",".join(map(str, tcn_cfg.get("dilations", [])))},
        {"Component": temporal_name, "Parameter": "kernel_size", "Value": tcn_cfg.get("kernel_size")},
        {"Component": temporal_name, "Parameter": "dropout", "Value": tcn_cfg.get("dropout")},
        {"Component": temporal_name, "Parameter": "expansion_ratio", "Value": tcn_cfg.get("expansion_ratio")},
        {"Component": temporal_name, "Parameter": "pool", "Value": tcn_cfg.get("pool")},
        {"Component": temporal_name, "Parameter": "batch_size", "Value": tcn_cfg.get("batch_size")},
        {"Component": temporal_name, "Parameter": "inference_batch_size", "Value": tcn_cfg.get("inference_batch_size")},
        {"Component": temporal_name, "Parameter": "epochs", "Value": tcn_cfg.get("epochs")},
        {"Component": temporal_name, "Parameter": "lr", "Value": tcn_cfg.get("lr")},
        {"Component": temporal_name, "Parameter": "weight_decay", "Value": tcn_cfg.get("weight_decay")},
        {"Component": "GraphAD+", "Parameter": "corr_threshold", "Value": graphad_cfg.get("corr_threshold")},
        {"Component": "GraphAD+", "Parameter": "alpha", "Value": graphad_cfg.get("alpha")},
        {"Component": "GraphAD+", "Parameter": "top_k", "Value": graphad_cfg.get("top_k")},
        {"Component": "GraphAD+", "Parameter": "lambda_z", "Value": graphad_cfg.get("lambda_z")},
        {"Component": "GraphAD+", "Parameter": "lambda_tr", "Value": graphad_cfg.get("lambda_tr")},
        {"Component": "GraphAD+", "Parameter": "lambda_fl", "Value": graphad_cfg.get("lambda_fl")},
        {"Component": "UTAR", "Parameter": "q_grid", "Value": ",".join(f"{q:.2f}" for q in routing_cfg.get("q_grid", []))},
        {"Component": "UTAR", "Parameter": "sigmoid_gain", "Value": routing_cfg.get("sigmoid_gain")},
        {"Component": "UTAR", "Parameter": "entropy_shortcut_quantile", "Value": routing_cfg.get("entropy_shortcut_quantile")},
        {"Component": "UTAR", "Parameter": "discrepancy_shortcut_quantile", "Value": routing_cfg.get("discrepancy_shortcut_quantile")},
    ]
    df = pd.DataFrame(rows)
    df = df[df["Value"].notna()].reset_index(drop=True)
    write_csv(METRIC_DIR / "table1_model_configuration.csv", df)
    print(df)


if __name__ == "__main__":
    main()
