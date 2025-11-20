"""
Data loader supporting sklearn toy datasets and CSV files.
Returns X (2D), y (1D), feature_names, target_names
"""
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import pandas as pd
import numpy as np

def load_dataset(name: str = "breast_cancer", csv_path: str = None, target_col: str = None):
    name = name.lower()
    if csv_path:
        df = pd.read_csv(csv_path)
        if target_col is None:
            raise ValueError("When csv_path provided, you must pass --target_col")
        y = df[target_col].values
        X = df.drop(columns=[target_col]).values
        feature_names = list(df.drop(columns=[target_col]).columns)
        return X, y, feature_names, None
    if name == "breast_cancer":
        d = load_breast_cancer()
    elif name == "iris":
        d = load_iris()
    elif name == "wine":
        d = load_wine()
    else:
        raise ValueError(f"Unknown dataset {name}")
    X = d.data
    y = d.target
    return X, y, list(d.feature_names), getattr(d, "target_names", None)
