import yaml
from src.data.loader import load_dataset
from src.models.xgb_wrapper import build_xgb
from src.models.lgb_wrapper import build_lgb
from src.models.io import save_model

def train(config_path='src/config/config.yaml'):
    cfg = yaml.safe_load(open(config_path))
    X_train, X_test, y_train, y_test = load_dataset()

    model_type = cfg["model"]
    params = cfg["params"][model_type[:3]] if model_type == "xgboost" else cfg["params"]["lgb"]

    model = build_xgb(params) if model_type == "xgboost" else build_lgb(params)
    model.fit(X_train, y_train)

    save_model(model, f"model_{model_type}.pkl")
    return model
