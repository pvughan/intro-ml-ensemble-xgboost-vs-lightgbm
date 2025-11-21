# Focused Comparative Study: XGBoost vs LightGBM

---

## 1. Introduction

This project provides a complete baseline for conducting a **Focused Comparative Study** between **XGBoost** and **LightGBM** on a standard classification problem — the Breast Cancer dataset from scikit-learn.

Included components:

- **Formulation** of the research problem
- **Model implementation** using official APIs
- **Evaluation** across multiple metrics
- **Ablation study** analyzing hyperparameter sensitivity
- **Structured modular codebase**
- **Visualization** of comparative results

This baseline can serve as a foundation for future benchmarking, extensions, or machine learning research projects.

---

## 2. Project Structure

```bash
project/
│── data/
│ └── load_data.py
│
│── src/
│ ├── formulation/
│ │ └── formulation.py
│ │
│ ├── models/
│ │ ├── xgboost_model.py
│ │ └── lightgbm_model.py
│ │
│ ├── evaluation/
│ │ └── evaluate.py
│ │
│ └── ablation/
│ └── ablation.py
│
│── main.py
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

## 4. Run the Baseline

#### Run the Notebook

```bash
jupyter notebook main.ipynb
```

#### Run the Python Script

```bash
python main.py
```

This runs:

1. Data loading
2. Problem formulation
3. XGBoost training
4. LightGBM training
5. Evaluation & comparisons
6. Ablation study
7. Visualization

## 5. Outputs

1. Metrics Table

- Accuracy
- Precision
- Recall
- F1 Score

2. Visual Charts

- Performance comparison bar chart
- Ablation line chart

3. Text Summary

- Which model performs better
- Model stability under hyperparameter variations

## 6. Module Breakdown

`data/load_data.py`  
Loads, scales, and splits the dataset.

`src/formulation/formulation.py`  
Defines the research hypothesis, evaluation criteria, and the comparison objective.

`src/models/`  
| File | Description |
| ------------------- | ---------------------------- |
| `xgboost_model.py` | XGBoost baseline classifier |
| `lightgbm_model.py` | LightGBM baseline classifier |

`src/evaluation/evaluate.py`  
Computes metrics and plots comparison charts.

`src/ablation/ablation.py`  
Runs controlled ablation on `max_depth`.
