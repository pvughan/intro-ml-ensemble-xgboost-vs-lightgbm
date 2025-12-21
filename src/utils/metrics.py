from typing import Dict, Any
import pandas as pd


def print_metrics(title: str, metrics: Dict[str, Any], precision: int = 4):
    """Print metrics in a formatted way.

    Args:
        title: Title for the metrics section.
        metrics: Dictionary of metric names and values.
        precision: Number of decimal places to display.
    """
    print(f"\n--- {title} ---")
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            if abs(v) < 1e-10:
                print(f"{k}: {v:.{precision}e}")
            else:
                print(f"{k}: {v:.{precision}f}")
        else:
            print(f"{k}: {v}")


def compare_models(results_dict: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Compare multiple models' metrics in a DataFrame.

    Args:
        results_dict: Dictionary mapping model names to their metrics.

    Returns:
        pandas.DataFrame with models as rows and metrics as columns.
    """
    return pd.DataFrame(results_dict).T


def get_best_metric(results_dict: Dict[str, Dict[str, float]], metric: str) -> tuple:
    """Get the model with the best value for a given metric.

    Args:
        results_dict: Dictionary mapping model names to their metrics.
        metric: Name of the metric to compare.

    Returns:
        Tuple of (model_name, best_value).
    """
    best_model = None
    best_value = float('-inf')
    
    for model_name, metrics in results_dict.items():
        if metric in metrics:
            value = metrics[metric]
            if value > best_value:
                best_value = value
                best_model = model_name
    
    return (best_model, best_value) if best_model else (None, None)
