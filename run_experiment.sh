#!/usr/bin/env bash
set -e
python src/train.py --model xgb --dataset breast_cancer --output results/xgb_baseline.joblib --cv
python src/train.py --model lgb --dataset breast_cancer --output results/lgb_baseline.joblib --cv
python src/evaluate.py --model-path results/xgb_baseline.joblib --dataset breast_cancer
python src/evaluate.py --model-path results/lgb_baseline.joblib --dataset breast_cancer
python src/ablation.py --model xgb --dataset breast_cancer --out results/ablation_xgb.csv
python src/ablation.py --model lgb --dataset breast_cancer --out results/ablation_lgb.csv
