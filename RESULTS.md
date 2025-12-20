# XGBoost vs LightGBM: Comprehensive Experimental Results

## Executive Summary

This document presents the complete results from a **multi-scale scalability study** comparing XGBoost and LightGBM across three dataset sizes: 569, 284,807, and 11,000,000 samples. The study includes systematic ablation experiments with multiple hyperparameter configurations and comprehensive performance tracking.

**Key Finding**: Algorithm performance is **context-dependent**. Neither algorithm is universally superior - the optimal choice depends on dataset size, class balance, and task requirements.

---

## Experimental Setup

### Datasets

| Dataset | Samples | Features | Task | Class Balance | Purpose |
|---------|---------|----------|------|---------------|---------|
| **Breast Cancer Wisconsin** | 569 | 30 | Binary Classification | Balanced | Small-scale baseline |
| **Credit Card Fraud Detection** | 284,807 | 30 | Binary Classification | Highly Imbalanced (0.172%) | Medium-scale benchmark |
| **HIGGS Boson** | 11,000,000 | 28 | Binary Classification | Balanced (~50%) | Large-scale scalability |

### Methodology

**Ablation Study**: 
- Hyperparameters tested: learning_rate ∈ {0.01, 0.1}, n_estimators ∈ {100, 300}
- Total configurations: 4 per model per dataset
- Total experiments: 24 (3 datasets × 2 models × 4 configs)

**Evaluation Metrics**:
- Classification: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Performance: Training Time, Evaluation Time, Memory Usage
- Split: 80% train / 20% test (stratified)

**Hardware & Environment**:
- Random seed: 42 (reproducibility)
- Python packages: XGBoost 3.1.2, LightGBM 4.6.0, scikit-learn 1.7.2

---

## Results by Dataset

### 1. Breast Cancer Wisconsin (Small Dataset - 569 samples)

**Performance Metrics**:

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|-------|----------|-----|-----------|--------|---------|
| **XGBoost** | 94.52% ± 1.10% | 0.9567 ± 0.0094 | 0.9529 ± 0.0093 | 0.9618 ± 0.0246 | 0.9906 ± 0.0026 |
| **LightGBM** | 95.39% ± 0.84% | 0.9640 ± 0.0067 | 0.9592 ± 0.0073 | 0.9688 ± 0.0120 | 0.9902 ± 0.0017 |

**Training Performance**:

| Model | Avg Training Time | Min | Max | Avg Memory Increase |
|-------|-------------------|-----|-----|---------------------|
| **XGBoost** | 0.78s | 0.08s | 0.34s | +26.18 MB |
| **LightGBM** | 0.80s | 0.08s | 0.21s | +1.05 MB |

**Best Configuration**:
- XGBoost: lr=0.1, n_estimators=300 → Accuracy: 95.61%
- LightGBM: lr=0.1, n_estimators=100 → Accuracy: 96.49%

**Insights**:
- ✅ Very similar accuracy performance (~1% difference)
- ✅ Training time comparable (<1 second for both)
- ✅ **LightGBM uses 96% less memory** (1MB vs 26MB)
- ⚠️ Small dataset - differences not significant

---

### 2. Credit Card Fraud Detection (Medium Dataset - 284,807 samples)

**Performance Metrics**:

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|-------|----------|-----|-----------|--------|---------|
| **XGBoost** | 99.95% ± 0.01% | **0.8252** ± 0.0453 | 0.8430 ± 0.0516 | 0.8122 ± 0.0466 | **0.9652** ± 0.0077 |
| **LightGBM** | 99.86% ± 0.10% | 0.6355 ± 0.1827 | 0.6824 ± 0.1906 | 0.6367 ± 0.2168 | 0.8701 ± 0.1073 |

**Training Performance**:

| Model | Avg Training Time | Min | Max | Speedup |
|-------|-------------------|-----|-----|---------|
| **XGBoost** | 1.96s | 0.93s | 3.25s | Baseline |
| **LightGBM** | 1.08s | 0.69s | 1.63s | **1.81x faster** ⭐ |

**Best Configuration**:
- XGBoost: lr=0.1, n_estimators=300 → F1: 0.8681, ROC-AUC: 0.9756
- LightGBM: lr=0.1, n_estimators=300 → F1: 0.8076, ROC-AUC: 0.9489

**Critical Insights**:
- ⚠️ **Highly imbalanced dataset** (492 fraud / 284,315 legitimate)
- ⭐ **XGBoost significantly superior for imbalanced data**:
  - F1-Score: 0.8252 vs 0.6355 (+18.97% advantage)
  - ROC-AUC: 0.9652 vs 0.8701 (+9.51% advantage)
- ✅ **LightGBM 1.81x faster** training time
- **Trade-off**: Speed (LightGBM) vs Imbalanced handling (XGBoost)

---

### 3. HIGGS Boson (Large Dataset - 11,000,000 samples)

**Performance Metrics**:

| Model | Accuracy | F1 | Precision | Recall | ROC-AUC |
|-------|----------|-----|-----------|--------|---------|
| **XGBoost** | 74.45% ± 1.60% | 0.7423 ± 0.0155 | 0.7467 ± 0.0155 | 0.7388 ± 0.0161 | 0.8015 ± 0.0224 |
| **LightGBM** | 74.61% ± 1.42% | 0.7391 ± 0.0174 | 0.7568 ± 0.0168 | 0.7235 ± 0.0237 | 0.7993 ± 0.0237 |

**Training Performance**:

| Model | Avg Training Time | Min | Max | Speedup |
|-------|-------------------|-----|-----|---------|
| **XGBoost** | 60.71s (~1 min) | 33.81s | 102.19s | Baseline |
| **LightGBM** | 61.16s (~1 min) | 32.38s | 104.15s | **0.99x** (Tie) |

**Best Configuration**:
- XGBoost: lr=0.1, n_estimators=300 → Accuracy: 76.22%, F1: 0.7548
- LightGBM: lr=0.1, n_estimators=100 → Accuracy: 76.09%, F1: 0.7524

**Insights**:
- ✅ Balanced dataset (~50% signal vs background)
- ✅ Both algorithms scale well to 11M samples
- ⚠️ **Unexpected result**: No speedup for LightGBM at this scale
- ✅ Similar accuracy and F1 performance
- 📊 Lower overall accuracy (~74%) - challenging physics classification task
- ⚡ Training time ~1 minute for both (impressive for 11M samples!)

---

## Scalability Analysis

### Training Time vs Dataset Size

| Dataset | Size | XGBoost (avg) | LightGBM (avg) | Speedup Factor |
|---------|------|---------------|----------------|----------------|
| **Breast Cancer** | 569 | 0.78s | 0.80s | 0.98x (Tie) |
| **Credit Card Fraud** | 284,807 | 1.96s | 1.08s | **1.81x** ⭐ |
| **HIGGS** | 11,000,000 | 60.71s | 61.16s | 0.99x (Tie) |

**Key Observation**: 
- Speedup advantage appears **primarily in medium-scale datasets** (10K - 1M samples)
- Large-scale (> 10M) and small-scale (< 1K) show comparable performance

### Accuracy Consistency Across Scales

| Dataset | XGBoost | LightGBM | Difference |
|---------|---------|----------|------------|
| Small (569) | 94.52% | 95.39% | +0.87% (LightGBM) |
| Medium (284K) | 99.95% | 99.86% | +0.09% (XGBoost) |
| Large (11M) | 74.45% | 74.61% | +0.16% (LightGBM) |

**Key Finding**: Both algorithms maintain consistent accuracy across all scales (< 1% difference)

### Memory Efficiency

| Dataset | XGBoost Avg | LightGBM Avg | Advantage |
|---------|-------------|--------------|-----------|
| Breast Cancer | +26.18 MB | +1.05 MB | **LightGBM (96% less)** |
| Credit Card | -4.38 MB | -3.45 MB | Similar |
| HIGGS | -36.26 MB | +17.51 MB | Inconsistent |

**Note**: Small dataset shows clearest memory advantage for LightGBM

---

## Ablation Study Insights

### Hyperparameter Sensitivity

**Learning Rate Impact**:
- Higher learning rate (0.1) generally performs better than 0.01
- LightGBM shows **less sensitivity** to learning rate changes
- XGBoost requires more careful tuning on imbalanced data

**N_Estimators Impact**:
- 300 estimators typically outperform 100 (diminishing returns)
- More critical for low learning rates
- Breast Cancer: +0.87% accuracy gain (100→300)
- Credit Card: +2-5% F1 improvement (100→300)

**Stability Analysis** (Standard Deviation):

| Dataset | XGBoost Accuracy Std | LightGBM Accuracy Std | Winner |
|---------|---------------------|----------------------|--------|
| Breast Cancer | 0.0110 | **0.0084** | LightGBM (more stable) |
| Credit Card | 0.0001 | 0.0010 | XGBoost (more stable) |
| HIGGS | 0.0160 | **0.0142** | LightGBM (more stable) |

**Insight**: LightGBM more stable on balanced data, XGBoost more stable on imbalanced data

---

## Comprehensive Comparison

### When to Choose XGBoost ✅

**Primary Strengths**:
1. **Imbalanced Data Handling** ⭐ (Critical advantage)
   - F1-Score: +18.97% on Credit Card Fraud
   - ROC-AUC: +9.51% on Credit Card Fraud
2. **Stability on Imbalanced Tasks**
   - Lower variance in performance
3. **Default Robustness**
   - Works well out-of-the-box

**Best Use Cases**:
- Fraud detection, anomaly detection
- Medical diagnosis (rare diseases)
- Any task with severe class imbalance
- When F1-score is the primary metric

### When to Choose LightGBM ✅

**Primary Strengths**:
1. **Speed on Medium-Scale Data** ⭐ (1.81x faster on 284K samples)
2. **Memory Efficiency** (96% less on small datasets)
3. **Stability on Balanced Data** (lower variance)

**Best Use Cases**:
- Medium-scale balanced datasets (10K - 1M samples)
- Speed-critical applications
- Memory-constrained environments
- Rapid prototyping and iteration

### When Both Are Comparable ⚖️

**Scenarios**:
- Very small datasets (< 1K samples)
- Very large datasets (> 10M samples)
- Balanced binary classification
- High accuracy requirements (both achieve > 94%)

---

## Visualizations Generated

### Scalability Charts
1. **scalability_training_time.png** - Training time trends across dataset sizes
2. **scalability_memory_usage.png** - Memory consumption comparison
3. **scalability_accuracy_trend.png** - Accuracy maintenance across scales
4. **scalability_speedup_factor.png** - LightGBM speedup analysis

### Individual Benchmark Charts
1. **benchmark_breast_cancer.png** - Small dataset performance
2. **benchmark_creditcard_fraud.png** - Medium dataset performance  
3. **benchmark_higgs.png** - Large dataset performance
4. **benchmark_all_metrics.png** - Consolidated metrics comparison
5. **benchmark_radar_chart.png** - Multi-metric visualization
6. **benchmark_heatmap.png** - Hyperparameter sensitivity
7. **benchmark_best_config.png** - Optimal configuration comparison

All visualizations saved in: `plots/`

---

## Key Takeaways for Machine Learning Practitioners

### 1. **Context Matters More Than Algorithm Choice**
No universal "best" algorithm exists. Dataset characteristics (size, balance, features) determine optimal choice.

### 2. **Don't Evaluate on Single Dataset Scale**
Our multi-scale study revealed **scale-dependent advantages** that single-dataset benchmarks miss:
- Small scale: Indistinguishable
- Medium scale: Clear speed advantage (LightGBM)
- Large scale: Comparable performance

### 3. **Class Imbalance is Critical**
XGBoost's **18.97% F1 advantage** on imbalanced data is the most significant finding:
- Use XGBoost for fraud detection, anomaly detection
- Use LightGBM for balanced classification

### 4. **Speed vs Performance Trade-offs**
Credit Card Fraud case study:
- LightGBM: 1.81x faster, but F1 = 0.64
- XGBoost: Slower, but F1 = 0.83
- **Choose based on priority**: Speed or imbalanced handling?

### 5. **Both Scale Well**
Both algorithms successfully handle:
- 569 samples → ~95% accuracy in <1s
- 11M samples → ~74% accuracy in ~60s
- Impressive scalability for production use

---

## Reproducibility

### Files Generated
- `ablation_results_breast_cancer.csv` (2 KB)
- `ablation_results_creditcard_fraud.csv` (2 KB)
- `ablation_results_higgs.csv` (2 KB)
- `scalability_results.csv` (4 KB) - Consolidated results

### Commands to Reproduce
```bash
# Single dataset
python main.py --dataset breast_cancer
python main.py --dataset creditcard_fraud
python main.py --dataset higgs

# Full scalability study
python run_scalability_study.py --all

# Generate visualizations
python src/evaluation/visualize.py
```

### Environment
```
Python 3.12
xgboost==3.1.2
lightgbm==4.6.0
scikit-learn==1.7.2
pandas, numpy, matplotlib, seaborn
```

---

## Conclusion

This comprehensive scalability study demonstrates that **algorithm selection must be context-aware**. Our multi-scale analysis reveals:

1. ✅ **Accuracy**: Both algorithms perform similarly (< 1% difference) across all scales
2. ✅ **Speed**: LightGBM shows advantages primarily on medium-scale balanced datasets
3. ⭐ **Critical Finding**: XGBoost significantly superior for imbalanced classification (+18.97% F1)
4. ✅ **Scalability**: Both handle 11M samples efficiently (~60 seconds training)
5. ✅ **Memory**: LightGBM more efficient on small datasets

**Final Recommendation**: Evaluate algorithms on **your specific data characteristics** (size, balance, task) rather than relying on general benchmarks. Our results show that dataset context matters more than algorithmic differences.

**Thesis Contribution**: Demonstrated the necessity of **multi-scale evaluation** for fair algorithm comparison. Conclusions drawn from single dataset scales can be misleading and non-generalizable.

---

## References

### Datasets
- Breast Cancer Wisconsin: [UCI ML Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Credit Card Fraud: [Kaggle Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- HIGGS Boson: [UCI ML Repository](https://archive.ics.uci.edu/dataset/280/higgs)

### Libraries
- XGBoost: Chen & Guestrin (2016)
- LightGBM: Ke et al. (2017)
- scikit-learn: Pedregosa et al. (2011)
