# Focused Comparative Study: XGBoost vs LightGBM

---

## 1. Introduction

This project conducts a **focused comparative study** between two widely-used gradient boosting frameworks: **XGBoost** and **LightGBM**, under a controlled experimental setting.

The objective is to analyze and compare their performance, stability, and sensitivity to hyperparameters on a standard binary classification task using the **Breast Cancer dataset** from scikit-learn.

Key components include:

- Clear problem formulation and comparison objective
- Model implementation using official APIs
- **Comprehensive evaluation using 5 different performance benchmarks**
- **Systematic hyperparameter ablation study across multiple configurations**
- Structured and reusable codebase
- Reproducible experimental results with visualizations

---

## Problem formulation (explicit)

We study **binary classification** with training data \(\{(x_i, y_i)\}_{i=1}^n\), where \(y_i \in \{0,1\}\). Both XGBoost and LightGBM learn an additive ensemble of decision trees (gradient boosting) by minimizing a regularized empirical risk.

- **Objective (typical)**: logloss (binary cross-entropy)
- **Prediction**: probability via a logistic link
- **Comparison protocol**: same dataset split (fixed seed) and same evaluation metrics across models

---

## 2. Project Structure

```bash
intro-ml-ensemble-xgboost-vs-lighgbm/
│── src/
│ ├── data/
│ │ └── loader.py
│ │
│ ├── models/
│ │ ├── xgboost_model.py
│ │ └── lightgbm_model.py
│ │
│ ├── evaluation/
│ │ ├── evaluator.py
│ │ ├── ablation.py
│ │ └── visualize.py
│ │
│ ├── utils/
│ │ └── metrics.py
│ │
│ └── ablation_results.csv
│
│── plots/                        # Benchmark visualization outputs
│ ├── benchmark_all_metrics.png
│ ├── benchmark_radar_chart.png
│ ├── benchmark_heatmap.png
│ └── benchmark_best_config.png
│
│── main.py
│── main.ipynb
│── requirements.txt
└── README.md
```

---

## 3. Installation

### Create environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Project

### Script-based Execution (Recommended)

Run the complete pipeline:

```bash
python main.py
```

This executes:
1. Data loading and preprocessing
2. Training XGBoost and LightGBM models
3. Model evaluation across 5 metrics
4. Ablation study on hyperparameters

### Notebook-based Execution (Optional)

For interactive exploration:

```bash
jupyter notebook main.ipynb
```

---

## 5. Evaluation

Both models are evaluated using **5 performance benchmarks**: Accuracy, Precision, Recall, F1-score, and ROC-AUC.

### Visualization

Generate benchmark charts:

```bash
python src/evaluation/visualize.py
```

This creates 4 charts in `plots/`:
- Metrics comparison across hyperparameter settings
- Radar chart for average performance
- Heatmap for hyperparameter sensitivity
- Best configuration comparison

---

## 6. Ablation Study

Systematic hyperparameter ablation study benchmarks both methods across different configurations.

**Hyperparameter Search Space:**
- Learning rate: [0.01, 0.05, 0.1]
- Number of estimators: [100, 300, 600]
- Subsample: [0.8, 1.0]
- Model-specific: max_depth/num_leaves, colsample_bytree

Each configuration is evaluated using all 5 performance benchmarks. Results are saved to `src/ablation_results.csv`.

## 7. Reproducibility

All experiments use fixed random seeds and shared data splits. Running `main.py` will reproduce the reported evaluation results.
