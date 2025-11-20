from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, log_loss
import numpy as np

def compute_classification_metrics(y_true, y_pred, y_prob=None):
    metrics = {}
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['f1_macro'] = float(f1_score(y_true, y_pred, average='macro'))
    # only compute roc_auc for binary with probs
    try:
        if y_prob is not None:
            if y_prob.ndim == 2 and y_prob.shape[1] >= 2:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob[:,1]))
            else:
                metrics['roc_auc'] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics['roc_auc'] = None
    return metrics
