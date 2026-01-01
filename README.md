# Multi-Scale Comparative Study: XGBoost vs LightGBM

> **Scalability Analysis Across Small, Medium, and Large Datasets**

---

## 1. Introduction

This project conducts a **comprehensive comparative study** between two widely-used gradient boosting frameworks: **XGBoost** and **LightGBM**, with focus on **scalability analysis across different dataset sizes**.

**Key Innovation**: Unlike traditional comparisons on a single dataset, this study examines how these algorithms perform across **three dataset scales**:

- **Small** (569 samples) - Breast Cancer Wisconsin
- **Medium** (~284K samples) - Credit Card Fraud Detection
- **Large** (11M samples) - HIGGS Boson

**Core Analysis:**

- Performance comparison (accuracy, F1, precision, recall, ROC-AUC)
- **Training time scalability**
- **Memory usage patterns**
- **Speedup factor analysis**
- Hyperparameter sensitivity across scales

---

## 2. Available Datasets

| Dataset               | Size       | Features | Task                  | Balance                    | Source  |
| --------------------- | ---------- | -------- | --------------------- | -------------------------- | ------- |
| **Breast Cancer**     | 569        | 30       | Binary Classification | Balanced                   | sklearn |
| **Credit Card Fraud** | 284,807    | 30       | Binary Classification | Highly Imbalanced (0.172%) | Kaggle  |
| **HIGGS Boson**       | 11,000,000 | 28       | Binary Classification | Balanced (~50%)            | UCI     |

### Dataset Details

**Breast Cancer Wisconsin (Small)**

- Diagnostic features from breast mass FNA
- Perfect for baseline comparison
- Fast experimentation

**Credit Card Fraud Detection (Medium)**

- European cardholders transactions (Sept 2013)
- PCA-transformed features (confidentiality)
- Tests model handling of severe class imbalance
- **Auto-download**: Requires Kaggle API credentials

**HIGGS Boson (Large)**

- Simulated particle physics events from LHC at CERN
- 11M samples, ~7.5GB compressed
- Demonstrates true scalability differences
- **Auto-download**: ~7.5GB from UCI repository
- **Recommendation**: Use subset for testing (e.g., 100K samples)

---

## 3. Installation

### Create Environment

```bash
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Setup for Credit Card Dataset (Optional)

If you plan to use the Credit Card Fraud dataset:

1. Install Kaggle API: (already in requirements.txt)
2. Setup credentials: Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<username>\.kaggle\` (Windows)
3. Get API key from: https://www.kaggle.com/settings

**Alternative**: Download manually and place in `data/raw/creditcard.csv`

---

## 4. Quick Start

### Single Dataset Analysis

Run on default dataset (Breast Cancer):

```bash
python main.py
```

Run on specific dataset:

```bash
# Small dataset
python main.py --dataset breast_cancer

# Medium dataset
python main.py --dataset creditcard_fraud

# Large dataset (WARNING: requires 16GB+ RAM and takes hours!)
python main.py --dataset higgs
```

### Multi-Dataset Scalability Study

**Recommended** - Small + Medium datasets:

```bash
python run_scalability_study.py --datasets breast_cancer creditcard_fraud
```

**Full scalability study** with HIGGS subset (recommended):

```bash
python run_scalability_study.py --all --higgs-sample 100000
```

**Complete study** (WARNING: full 11M HIGGS dataset!):

```bash
python run_scalability_study.py --all
```

List available datasets:

```bash
python run_scalability_study.py --list
```

---

## 5. Project Structure

```bash
intro-ml-ensemble-xgboost-vs-lightgbm/
├── data/
│   ├── raw/                    # Downloaded datasets (auto-created)
│   └── processed/              # Preprocessed data cache
│
├── src/
│   ├── config.py              # Dataset metadata and configurations
│   │
│   ├── data/
│   │   └── loader.py          # Multi-dataset loader with auto-download
│   │
│   ├── models/
│   │   ├── xgboost_model.py
│   │   └── lightgbm_model.py
│   │
│   ├── evaluation/
│   │   ├── evaluator.py       # Performance metrics
│   │   ├── ablation.py        # Ablation study with timing/memory tracking
│   │   ├── scalability_analysis.py  # Multi-dataset orchestration
│   │   └── visualize.py       # Scalability visualizations
│   │
│   └── utils/
│       └── metrics.py
│
├── plots/                      # Generated visualizations
│   ├── scalability_training_time.png
│   ├── scalability_memory_usage.png
│   ├── scalability_accuracy_trend.png
│   └── scalability_speedup_factor.png
│
├── results/                    # Experimental results
│   ├── ablation_results_breast_cancer.csv
│   ├── ablation_results_creditcard_fraud.csv
│   ├── ablation_results_higgs.csv
│   └── scalability_results.csv          # Consolidated results
│
├── main.py                     # Single dataset analysis
├── run_scalability_study.py   # Multi-dataset scalability study
├── main.ipynb                  # Interactive notebook
├── requirements.txt
└── README.md
```

---

## 6. Evaluation Methodology

### Performance Metrics

Five comprehensive benchmarks for each model:

1. **Accuracy** - Overall classification correctness
2. **Precision** - Positive prediction accuracy
3. **Recall** - True positive detection rate
4. **F1-Score** - Harmonic mean of precision and recall
5. **ROC-AUC** - Area under ROC curve

### Scalability Metrics (NEW)

For multi-dataset analysis, we additionally track:

- **Training Time** - Time to fit model on training data
- **Evaluation Time** - Time to predict on test data
- **Memory Usage** - Peak memory increase during training
- **Speedup Factor** - XGBoost time / LightGBM time

### Ablation Study

Systematic hyperparameter search:

- **Learning rate**: [0.01, 0.1]
- **Number of estimators**: [100, 300]
- **Total configurations**: 2 models × 4 configs = 8 experiments per dataset

---

## 7. Expected Results & Insights

### Performance Patterns by Scale

**Small Dataset (Breast Cancer - 569 samples)**

- Both algorithms perform similarly (~96% accuracy)
- Training time differences negligible (< 1 second)
- Difficult to distinguish algorithm strengths

**Medium Dataset (Credit Card - 284K samples)**

- LightGBM shows 2-3x faster training
- Similar accuracy (~99%+)
- Class imbalance handling becomes testable

**Large Dataset (HIGGS - 11M samples)**

- **LightGBM is 10-20x faster** than XGBoost
- Significant memory usage differences
- Scalability advantages clearly visible
- Accuracy differences become measurable

### Key Findings

✅ **LightGBM Advantages:**

- Faster training on medium-large datasets
- Lower memory consumption
- Better handling of high-cardinality features

✅ **XGBoost Advantages:**

- Comparable accuracy on all scales
- More mature ecosystem
- Better default regularization

---

## 8. Visualization

### Generate Scalability Plots

After running multi-dataset analysis:

```bash
python src/evaluation/visualize.py
```

**Generated visualizations:**

1. **Training Time vs Dataset Size** - Scalability curves (log-log plot)
2. **Memory Usage Comparison** - Memory consumption across datasets
3. **Accuracy Across Scales** - Performance trends
4. **Speedup Factor** - LightGBM speedup over XGBoost

---

## 9. Hardware Requirements

| Dataset             | RAM Required | Typical Training Time\* | Disk Space |
| ------------------- | ------------ | ----------------------- | ---------- |
| Breast Cancer       | 2GB          | <1 second               | Negligible |
| Credit Card Fraud   | 4GB          | 1-5 minutes             | 150MB      |
| HIGGS (100K subset) | 4GB          | 5-15 minutes            | 100MB      |
| HIGGS (full 11M)    | **16GB+**    | **2-6 hours**           | **7.5GB**  |

\*Times vary based on hardware and hyperparameters

**Recommendation**: Start with small+medium datasets, then optionally try HIGGS subset.

---

## 10. Reproducibility

All experiments use:

- Fixed random seeds (default: 42)
- Consistent train/test splits
- Stratified sampling (for imbalanced data)
- Identical hyperparameter search spaces

Running `main.py` or `run_scalability_study.py` will reproduce the reported results.

---

## 11. Citation & References

### Datasets

- **Breast Cancer**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Credit Card Fraud**: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **HIGGS**: [UCI ML Repository](https://archive.ics.uci.edu/dataset/280/higgs)

### Libraries

- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- scikit-learn: Pedregosa et al. (2011)

---

## 12. Troubleshooting

**Issue**: Credit Card dataset download fails

- **Solution**: Download manually from Kaggle and place in `data/raw/creditcard.csv`

**Issue**: HIGGS dataset too large / out of memory

- **Solution**: Use subset via `--higgs-sample 100000`

**Issue**: Kaggle API authentication error

- **Solution**: Follow https://www.kaggle.com/docs/api to setup credentials

---

## 13. Future Extensions

Potential enhancements:

- [ ] Add CatBoost for 3-way comparison
- [ ] Include regression datasets
- [ ] GPU acceleration benchmarks
- [ ] Hyperparameter optimization (Optuna/Hyperopt)
- [ ] Production deployment timing
- [ ] Distributed training comparison

---

## License

MIT License - Feel free to use for research and education.

## Contact

For questions or suggestions, please open an issue or submit a Pull Request.

---

**⭐ Key Takeaway**: This project demonstrates that **algorithm scalability cannot be assessed on small datasets alone**. LightGBM's advantages emerge clearly only at medium-large scales, making multi-dataset evaluation essential for fair comparison.
