from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import numpy as np


def load_data(
    test_size: float = 0.2, 
    random_state: int = 42,
    scale: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess the Breast Cancer dataset.

    Args:
        test_size: Proportion of dataset to include in test split.
        random_state: Random seed for reproducibility.
        scale: If True, apply StandardScaler to features.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).

    Raises:
        ValueError: If test_size is not in (0, 1).
    """
    if not 0 < test_size < 1:
        raise ValueError(f"test_size must be in (0, 1), got {test_size}")

    data = load_breast_cancer()
    X, y = data.data, data.target

    if scale:
        X = StandardScaler().fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return X_train, X_test, y_train, y_test
