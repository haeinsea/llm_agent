from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


LABEL_CANDIDATES = ["fault", "fault_raw", "label", "y", "y_true"]


def load_uploaded_te_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]
    return df


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    y = None
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            y = df[col]
            df = df.drop(columns=[col])
            break
    # Remove identifier-like columns such as id / index / simulationRun / sample.
    for c in ["id", "index", "simulationRun", "sample", "Unnamed: 0"]:
        
        if c in df.columns:
            df = df.drop(columns=[c])
    return df, y


def scale_features(X: pd.DataFrame) -> np.ndarray:
    scaler = StandardScaler()
    return scaler.fit_transform(X.values)

# def scale_features(X: pd.DataFrame) -> np.ndarray:
#     # The eval_te_routing pipeline does not apply feature scaling.
#     return X.values.astype(float)
