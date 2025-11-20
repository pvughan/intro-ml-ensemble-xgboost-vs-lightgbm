# Ensemble: XGBoost vs LightGBM — Baseline

This repository is a baseline for the project **"Ensemble: XGBoost vs LightGBM"**.
It follows the project specs:
- Formulation: Yes
- Method: Focused Comparative
- Impact: No
- Implement: API (a minimal Flask API is provided)
- Evaluation: Yes (CV and holdout; metrics saved)
- Ablation: Yes (hyperparameter sweep script)

## Contents
- `requirements.txt` — required Python packages
- `config.yaml` — baseline config
- `src/` — source code
  - `data.py` — dataset loaders and preprocessing
  - `models.py` — wrappers for XGBoost and LightGBM
  - `train.py` — CLI script to train & save a model
  - `evaluate.py` — evaluate a saved model / compute metrics
  - `ablation.py` — run ablation studies (sweep hyperparameters)
  - `api.py` — simple Flask API to train and predict
  - `utils.py` — common utilities (logging/results)
- `run_experiment.sh` — example run script
- `results/` — outputs (created when running)

## Quick start (local)
1. Create a virtualenv and install:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Train baseline XGBoost on breast_cancer dataset (holdout):
```bash
python src/train.py --model xgb --dataset breast_cancer --output results/xgb_baseline.joblib
```

3. Evaluate:
```bash
python src/evaluate.py --model-path results/xgb_baseline.joblib --dataset breast_cancer
```

4. Run ablation (example):
```bash
python src/ablation.py --model xgb --dataset breast_cancer --out results/ablation_xgb.csv
```

5. Run API (dev):
```bash
python src/api.py
# endpoints:
# POST /train  -> JSON: {"model":"xgb","dataset":"breast_cancer"}
# POST /predict -> JSON: {"model_path":"results/xgb_baseline.joblib","X":[[...],...]}
```

## Notes
- Datasets come from `sklearn.datasets` for reproducibility and simplicity (breast_cancer, wine, digits).
- The baseline focuses on comparing XGBoost and LightGBM using the same preprocessing and evaluation pipeline.
