from data.loader import load_dataset
from evaluation.cross_validation import cross_validate_model
from models.xgboost_model import get_xgboost_model
from models.lightgbm_model import get_lightgbm_model
import pandas as pd


# ===== BREAST CANCER (balanced dataset) =====
X_train, X_test, y_train, y_test = load_dataset("breast_cancer")

X = pd.concat([X_train, X_test])
y = pd.concat([y_train, y_test])

cross_validate_model(
    lambda: get_xgboost_model(),
    X, y,
    k=5,
    model_name="XGBoost_BreastCancer"
)

cross_validate_model(
    lambda: get_lightgbm_model(),
    X, y,
    k=5,
    model_name="LightGBM_BreastCancer"
)


# ===== CREDIT CARD (imbalanced dataset) =====
X_train, X_test, y_train, y_test = load_dataset("creditcard_fraud")

X = X_train.append(X_test)
y = y_train.append(y_test)

pos = sum(y == 1)
neg = sum(y == 0)
ratio = neg / pos


cross_validate_model(
    lambda: get_xgboost_model(params={"scale_pos_weight": ratio}),
    X, y,
    k=5,
    model_name="XGBoost_Fraud"
)

cross_validate_model(
    lambda: get_lightgbm_model(params={"class_weight": "balanced"}),
    X, y,
    k=5,
    model_name="LightGBM_Fraud"
)


print("\n✅ Cross-validation experiments completed!")