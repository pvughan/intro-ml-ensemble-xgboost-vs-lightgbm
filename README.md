# Focused Comparative Study: XGBoost vs LightGBM

---

## 1. Introduction

This project conducts a **focused comparative study** between two widely-used gradient boosting frameworks: **XGBoost** and **LightGBM**, under a controlled experimental setting.

The objective is to analyze and compare their performance, stability, and sensitivity to hyperparameters on a standard binary classification task using the **Breast Cancer dataset** from scikit-learn.

Key components include:
- Clear problem formulation and comparison objective
- Model implementation using official APIs
- Comprehensive evaluation using multiple performance metrics
- Ablation study to analyze hyperparameter sensitivity
- Structured and reusable codebase
- Reproducible experimental results
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
│ │ ├── run_evaluation.py
│ │ ├── ablation.py
│ │ └── run_ablation.py
│ │
│ ├── utils/
│ │ └── metrics.py
│ │
│ └── ablation_results.csv
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
This project provides **two execution entry points**, each serving a different purpose.

#### Option A: Notebook-based Execution (Exploration & Visualization)

The notebook `main.ipynb` is intended for **interactive exploration**, **step-by-step execution**, and **visual inspection** of model behavior.

It is useful for:
- Understanding the training pipeline
- Inspecting intermediate results
- Demonstration and explanation during presentations

Run the notebook using:

```bash
jupyter notebook main.ipynb
```

#### Option B: Script-based Execution (Reproducible Evaluation)

The script `main.py` runs the **entire pipeline end-to-end** in a fully reproducible manner and is intended to generate **final evaluation results** used in reporting.

This includes:
1.	Data loading and preprocessing
2.	Training XGBoost and LightGBM models
3.	Model evaluation across multiple metrics
4.	Comparative analysis
5.	Ablation study on selected hyperparameters

Run the script using:

```bash
python main.py
```
---

## 5. Evaluation

Model performance is evaluated using:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Results are presented via:
- Metric tables
- Comparison charts
- Ablation plots

A short qualitative discussion is included to compare model stability.

---

## 6. Ablation Study

An ablation study is conducted to analyze model sensitivity with respect to key hyperparameters:
- Learning rate
- Number of estimators

Each model is trained under different hyperparameter configurations, and performance metrics are recorded for comparison.

The ablation results are saved to:
`src/ablation_results.csv`

**Sample Ablation Output (format illustration)**

| Model | learning_rate | n_estimators | Accuracy | F1 | ROC-AUC |
|------|---------------|--------------|----------|----|---------|
| XGBoost | 0.01 | 100 | 0.965 | 0.972 | 0.990 |
| LightGBM | 0.10 | 300 | 0.965 | 0.972 | 0.992 |

## 7. Module Breakdown

### Data
`src/data/loader.py`  
Loads and preprocesses the Breast Cancer dataset.

### Models
| File | Description |
|------|------------|
| `xgboost_model.py` | XGBoost classifier wrapper |
| `lightgbm_model.py` | LightGBM classifier wrapper |

### Evaluation
`src/evaluation/evaluator.py`  
Computes classification metrics.

`src/evaluation/run_evaluation.py`  
Runs evaluation pipeline and prints metrics.

### Ablation
`src/evaluation/ablation.py`  
Defines hyperparameter ablation logic.

`src/evaluation/run_ablation.py`  
Executes ablation experiments and saves results.

## 8. Reproducibility

To ensure reproducibility:
- All experiments use fixed random seeds where applicable
- The same data splits are shared across models
- Hyperparameter search spaces are explicitly defined

Running `main.py` will reproduce the reported evaluation results.
