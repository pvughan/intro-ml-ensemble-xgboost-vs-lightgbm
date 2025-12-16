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
│ │ ├── run_ablation.py
│ │ └── visualize.py          # Benchmark visualization generator
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

### Multiple Benchmark Metrics

This project implements a **comprehensive multi-benchmark evaluation** to rigorously assess model performance. Both XGBoost and LightGBM are evaluated using **5 different performance benchmarks**:

1. **Accuracy** - Overall classification correctness
2. **Precision** - Positive prediction accuracy
3. **Recall** - True positive detection rate
4. **F1-score** - Harmonic mean of precision and recall
5. **ROC-AUC** - Area under the receiver operating characteristic curve

Each model is benchmarked across **multiple hyperparameter configurations** to ensure robust and comprehensive comparison.

### Benchmark Visualization

To generate visualization charts showcasing benchmark results:

```bash
python src/evaluation/visualize.py
```

This generates 4 comprehensive charts in the `plots/` directory:

1. **Metrics Comparison Chart** - All 5 benchmarks across different hyperparameter settings
2. **Radar Chart** - Average performance visualization across all benchmarks
3. **Heatmap** - Hyperparameter sensitivity analysis for both models
4. **Best Configuration** - Comparison of optimal settings for each model

Results are presented via:
- Detailed metric tables
- Interactive comparison charts
- Ablation study visualizations
- Statistical performance summaries

A qualitative discussion comparing model stability and hyperparameter sensitivity is included.

---

## 6. Ablation Study

### Systematic Hyperparameter Benchmarking

A comprehensive ablation study is conducted to **benchmark both methods on different settings of hyperparameters**. This systematic approach evaluates model sensitivity and optimal configuration discovery.

**Hyperparameter Search Space:**
- **Learning rate**: [0.01, 0.1]
- **Number of estimators**: [100, 300]

**Total Benchmark Experiments:** 2 models × 4 hyperparameter configurations = **8 complete evaluations**

Each configuration is evaluated using all 5 performance benchmarks (Accuracy, F1, Precision, Recall, ROC-AUC), providing a comprehensive 40-metric comparison matrix.

The complete ablation results are saved to:  
`src/ablation_results.csv`

**Sample Ablation Output (actual results)**

| Model | learning_rate | n_estimators | Accuracy | F1 | ROC-AUC |
|---------|---------------|--------------|----------|--------|----------|
| XGBoost | 0.01 | 100 | 0.9649 | 0.9722 | 0.9895 |
| LightGBM| 0.01 | 100 | 0.9561 | 0.9655 | 0.9921 |
| XGBoost | 0.10 | 300 | 0.9561 | 0.9650 | 0.9934 |
| LightGBM| 0.10 | 300 | 0.9649 | 0.9722 | 0.9921 |

This systematic benchmarking approach enables:
- Identification of optimal hyperparameter configurations
- Analysis of model stability across different settings
- Fair comparison under controlled experimental conditions

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

### Ablation & Visualization
`src/evaluation/ablation.py`  
Defines hyperparameter ablation logic and benchmark configurations.

`src/evaluation/run_ablation.py`  
Executes ablation experiments and saves results.

`src/evaluation/visualize.py`  
Generates comprehensive benchmark visualization charts across all metrics and hyperparameter settings.

## 8. Reproducibility

To ensure reproducibility:
- All experiments use fixed random seeds where applicable
- The same data splits are shared across models
- Hyperparameter search spaces are explicitly defined

Running `main.py` will reproduce the reported evaluation results.
