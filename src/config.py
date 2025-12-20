"""
Configuration file for multi-scale dataset experiments
Contains dataset metadata, URLs, and hyperparameter configurations
"""

import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
PLOTS_DIR = PROJECT_ROOT / "plots"

# Create directories if they don't exist
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, RESULTS_DIR, PLOTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Dataset configurations
DATASETS = {
    "breast_cancer": {
        "name": "Breast Cancer Wisconsin",
        "source": "sklearn",
        "task": "binary_classification",
        "samples": 569,
        "features": 30,
        "description": "Diagnostic breast cancer dataset",
        "class_balance": "balanced",
        "url": None,  # Built-in dataset
    },
    "creditcard_fraud": {
        "name": "Credit Card Fraud Detection",
        "source": "kaggle",
        "task": "binary_classification", 
        "samples": 284807,
        "features": 30,
        "description": "European cardholders transactions (PCA-transformed)",
        "class_balance": "highly_imbalanced (0.172% fraud)",
        "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "kaggle_dataset": "mlg-ulb/creditcardfraud",
        "filename": "creditcard.csv",
        "target_column": "Class",
    },
    "higgs": {
        "name": "HIGGS Boson",
        "source": "uci",
        "task": "binary_classification",
        "samples": 11000000,
        "features": 28,
        "description": "Simulated particle physics events from LHC",
        "class_balance": "balanced (~50% signal)",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz",
        "filename": "HIGGS.csv.gz",
        "target_column": 0,  # First column is target
    },
}

# Default hyperparameter search space for ablation study
ABLATION_SEARCH_SPACE = {
    "learning_rate": [0.01, 0.1],
    "n_estimators": [100, 300],
}

# Default model parameters
DEFAULT_XGBOOST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.9,
    "random_state": 42,
}

DEFAULT_LIGHTGBM_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "subsample": 0.9,
    "random_state": 42,
    "verbose": -1,  # Suppress warnings
}

# Memory and performance settings
MEMORY_LIMIT_GB = 8  # Warning threshold for dataset loading
CHUNK_SIZE = 100000  # For large dataset processing

# Random seed for reproducibility
RANDOM_STATE = 42
