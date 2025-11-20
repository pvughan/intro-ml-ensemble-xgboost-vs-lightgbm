# Ensemble: XGBoost vs LightGBM — Baseline

This repository provides a baseline for comparing XGBoost and LightGBM on tabular classification tasks.
It includes:

- data loader for sklearn datasets (breast_cancer, iris, wine) and CSV
- training script using model APIs
- evaluation pipeline (cross-validation, metrics)
- ablation script (grid/random search)
- saving of model and results

## Quickstart

1. Create environment and install:

```bash
python -m venv .venv
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

2. Run baseline training + evaluation:

```bash
python src/training/train.py --dataset breast_cancer --model xgb --out_dir results/xgb_breast
python src/training/train.py --dataset breast_cancer --model lgb --out_dir results/lgb_breast
```

3. Run ablation (grid search example):

```bash
python src/ablation/ablation_runner.py --dataset wine --algo both --mode grid --out_dir results/ablation_wine
```

4. Evaluate saved model:

```bash
python src/evaluation/evaluate.py --model_path results/xgb_breast/model.joblib --dataset breast_cancer
```
