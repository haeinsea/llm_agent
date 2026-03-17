from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.io import read_yaml, write_csv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
METRIC_DIR = OUTPUT_DIR / "metrics"


def main() -> None:
    rf_cfg = read_yaml(CONFIG_DIR / "train_rf.yaml", default={})
    xgb_cfg = read_yaml(CONFIG_DIR / "train_xgb.yaml", default={})
    tcn_cfg = read_yaml(CONFIG_DIR / "train_tcn.yaml", default={})
    routing_cfg = read_yaml(CONFIG_DIR / "routing.yaml", default={})

    rows = [
        {"Component": "RF", "Parameter": "n_estimators", "Value": rf_cfg.get("n_estimators")},
        {"Component": "RF", "Parameter": "max_depth", "Value": rf_cfg.get("max_depth")},
        {"Component": "RF", "Parameter": "min_samples_leaf", "Value": rf_cfg.get("min_samples_leaf")},
        {"Component": "XGB", "Parameter": "n_estimators", "Value": xgb_cfg.get("n_estimators")},
        {"Component": "XGB", "Parameter": "max_depth", "Value": xgb_cfg.get("max_depth")},
        {"Component": "XGB", "Parameter": "learning_rate", "Value": xgb_cfg.get("learning_rate")},
        {"Component": "XGB", "Parameter": "subsample", "Value": xgb_cfg.get("subsample")},
        {"Component": "TCN", "Parameter": "window_size", "Value": tcn_cfg.get("window_size")},
        {"Component": "TCN", "Parameter": "channels", "Value": ",".join(map(str, tcn_cfg.get("channels", [])))},
        {"Component": "TCN", "Parameter": "kernel_size", "Value": tcn_cfg.get("kernel_size")},
        {"Component": "TCN", "Parameter": "dropout", "Value": tcn_cfg.get("dropout")},
        {"Component": "UTAR", "Parameter": "q_grid", "Value": ",".join(f"{q:.2f}" for q in routing_cfg.get("q_grid", []))},
        {"Component": "UTAR", "Parameter": "xgb_shortcut_low", "Value": routing_cfg.get("xgb_shortcut_low")},
        {"Component": "UTAR", "Parameter": "xgb_shortcut_high", "Value": routing_cfg.get("xgb_shortcut_high")},
    ]
    write_csv(METRIC_DIR / "table1_model_configuration.csv", pd.DataFrame(rows))
    print(pd.DataFrame(rows))


if __name__ == "__main__":
    main()
