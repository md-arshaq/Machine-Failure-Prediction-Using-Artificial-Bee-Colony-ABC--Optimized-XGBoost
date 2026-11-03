# Experimental Report: ABC-Optimized XGBoost for Machine Failure Prediction

**Course:** Bio-Inspired Algorithms (23DS7BSBIO)  
**Institution:** B.M.S. College of Engineering, Bengaluru  
**Dataset:** AI4I 2020 Predictive Maintenance Dataset (UCI Repository)  
**Target:** `Machine failure` (Imbalanced binary classification, ~3.39% positive)  

---

## 1. Project Overview & Problem Definition
Industrial machinery operates continuously under complex mechanical and thermal stresses. Sudden equipment failure causes catastrophic production downtime, costly reactive maintenance, and safety hazards. Predictive maintenance seeks to forecast impending failures from operating telemetry before catastrophic breakdown.

In this investigation:
1. An **XGBoost classification model** was developed to predict machine failure from sensor telemetry.
2. A **custom Artificial Bee Colony (ABC) optimizer** was developed from scratch in pure NumPy to jointly optimize:
   - **Feature Selection:** Identifying the minimal, most informative subset among 7 logical feature groups.
   - **Hyperparameter Optimization:** Tuning 8 critical regularization, capacity, and imbalance-weighting hyperparameters of XGBoost.
3. Strict **data leakage prevention** was enforced by stripping direct failure modes (`TWF`, `HDF`, `PWF`, `OSF`, `RNF`) and identifiers (`UDI`, `Product ID`).
4. Optimization was driven by **validation Average Precision (PR-AUC)** rather than misleading accuracy.

---

## 2. Preprocessing & Experimental Setup
- **Dataset Size:** 10,000 observations (9,661 normal operations, 339 failure events).
- **Partitioning:** Stratified 70% Train (7,000), 15% Validation (1,500), 15% Isolated Test (1,500).
- **Feature Engineering:** `temperature_difference = Process temperature [K] - Air temperature [K]`.
- **Encoding:** `OneHotEncoder` fitted strictly on training data for categorical feature `Type`.
- **Target Isolation:** The test partition was kept completely untouched during feature selection and optimization.

---

## 3. Discovered Optimal Solution via ABC
- **Optimal Validation Average Precision:** `0.8698`
- **Actual Model Evaluations:** `132`
- **Cache Hits:** `86`
- **Selected Feature Groups (6/7):**
  - Type (L, M, H)
  - Process temperature [K]
  - Rotational speed [rpm]
  - Torque [Nm]
  - Tool wear [min]
  - temperature_difference

### Optimal Hyperparameters:
- `n_estimators`: 561
- `max_depth`: 6
- `learning_rate`: 0.0587
- `min_child_weight`: 7.43
- `subsample`: 0.726
- `colsample_bytree`: 0.935
- `gamma`: 2.47
- `scale_pos_weight`: 4.36

---

## 4. Final Performance Comparison Across Multiple Seeds

### Table 7: Master Comparison Across 5 Seeds (Mean ± Std)

| Model                 | accuracy        | precision       | recall          | f1              | macro_f1        | roc_auc         | average_precision   |
|:----------------------|:----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:--------------------|
| Baseline XGBoost      | 0.9821 ± 0.0016 | 0.7118 ± 0.0297 | 0.8000 ± 0.0164 | 0.7530 ± 0.0176 | 0.8719 ± 0.0092 | 0.9750 ± 0.0019 | 0.8372 ± 0.0092     |
| ABC-Optimized XGBoost | 0.9839 ± 0.0014 | 0.7382 ± 0.0258 | 0.8157 ± 0.0107 | 0.7749 ± 0.0159 | 0.8832 ± 0.0083 | 0.9809 ± 0.0009 | 0.8481 ± 0.0040     |


---

## 5. Academic Discussion & Answers to Key Research Questions

1. **Why is XGBoost appropriate for predictive maintenance?**  
   XGBoost handles mixed tabular data, non-linear sensor thresholds, and missing values natively, while providing gradient-boosted decision boundaries that are resilient to feature scale variations.

2. **Why is accuracy insufficient as the only metric?**  
   With a 96.6% majority negative class, a naive classifier predicting zero failures achieves 96.6% accuracy but provides zero utility for maintenance planning. Precision-Recall AUC (Average Precision) and F1-score evaluate genuine minority failure detection.

3. **How does ABC explore the hyperparameter search space?**  
   Employed bees exploit known local neighborhoods via perturbation ($v_{ij} = x_{ij} + \phi_{ij}(x_{ij} - x_{kj})$), Onlooker bees focus search around high-nectar (high-AP) solutions via roulette-wheel selection, and Scout bees abandon stagnant basins when unimproved for $\ge 5$ trials to maintain global exploration.

4. **How does feature selection influence the model?**  
   Pruning noisy or redundant sensor measurements prevents tree overfitting and enhances generalization on rare failure signatures.

5. **Limitations:**  
   The AI4I 2020 dataset is synthetic (based on physical simulator equations). While it mirrors realistic sensor dynamics, real factory machinery exhibits unmodeled environmental noise, sensor degradation, and non-stationary operating cycles.

---

## 6. Generated Visualizations & Tables Summary
All 15 figures are stored in `results/figures/`:
- `fig01_target_distribution.png`: Class imbalance visualization.
- `fig02_correlation_heatmap.png`: Sensor correlation heatmap.
- `fig03_feature_distributions.png`: Density distributions by failure state.
- `fig04_failure_by_type.png`: Failure breakdown across product categories.
- `fig05_abc_convergence.png`: Best and population mean AP convergence.
- `fig06_best_fitness_vs_evals.png`: Fitness progression vs. evaluation count.
- `fig07_exploration_exploitation.png`: Employed improvements vs. scout resets.
- `fig08_metric_comparison.png`: Bar chart comparison with standard deviations.
- `fig09_baseline_confusion_matrix.png` & `fig10_optimized_confusion_matrix.png`: Test confusion matrices.
- `fig11_roc_curves.png` & `fig12_precision_recall_curves.png`: Discrimination curves.
- `fig13_feature_importance.png`: Gain-based feature importance of ABC model.
- `fig14_feature_selection_frequency.png`: Feature selection frequency across evaluations.
- `fig15_search_analysis.png`: Hyperparameter search landscape distribution.
