from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_base_routing_score(
    df: pd.DataFrame,
    cfg: dict[str, Any],
    rf_col: str = "p_rf",
    xgb_col: str = "p_xgb",
    tcn_col: str = "p_tcn",
) -> pd.Series:
    tcn_weight = float(cfg.get("tcn_weight", 1.0))
    tcn_weight = max(tcn_weight, 0.0)
    score = np.maximum.reduce(
        [
            df[rf_col].to_numpy(dtype=float),
            df[xgb_col].to_numpy(dtype=float),
            np.clip(df[tcn_col].to_numpy(dtype=float) * tcn_weight, 0.0, 1.0),
        ]
    )
    return pd.Series(score, index=df.index, name="p_utar_base")
