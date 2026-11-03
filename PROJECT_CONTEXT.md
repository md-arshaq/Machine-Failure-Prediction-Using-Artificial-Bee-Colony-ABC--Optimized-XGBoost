# PROJECT CONTEXT AND IMPLEMENTATION SPECIFICATION

# Machine Failure Prediction Using Artificial Bee Colony (ABC)-Optimized XGBoost

**Institution:** B.M.S. College of Engineering, Bengaluru  
**Department:** Computer Science and Engineering (Data Science)  
**Course:** Bio-Inspired Algorithms (23DS7BSBIO)  
**Project Type:** Academic Assignment / AAT  
**Academic Year:** 2026–2027  
**Primary Execution Environment:** Google Colab CPU / Local CPU  
**GPU Requirement:** Not required

---

# 1. PROJECT OVERVIEW

## 1.1 Project Title

**Machine Failure Prediction Using Artificial Bee Colony (ABC)-Optimized XGBoost with Joint Feature Selection and Hyperparameter Optimization**

## 1.2 Problem Statement

Industrial machinery operates under varying conditions involving temperature, rotational speed, torque, and tool wear. Unexpected machine failures can cause production downtime, financial losses, equipment damage, and safety risks.

Traditional maintenance strategies often rely on scheduled inspections or reactive maintenance after a failure has occurred.

Predictive maintenance uses machine operating data to identify potential failures before they happen.

The objective of this project is to develop a machine learning-based predictive maintenance system that predicts whether an industrial machine will experience a failure based on its operational characteristics.

An XGBoost classification model will serve as the underlying predictive model. Its performance will be improved using the Artificial Bee Colony (ABC) optimization algorithm.

The ABC algorithm will jointly optimize:

1. Feature selection: Identify the most useful operational features.
2. Hyperparameter optimization: Identify suitable XGBoost hyperparameters.

The optimized model will be compared against a baseline XGBoost model trained without bio-inspired optimization.

The study will evaluate whether ABC-based optimization improves predictive performance, particularly its ability to identify machine failures in an imbalanced classification dataset.

## 1.3 Project Objectives

The main objectives are:

1. Understand and preprocess the AI4I 2020 Predictive Maintenance dataset.
2. Perform exploratory data analysis to understand machine operating characteristics and failure patterns.
3. Develop a baseline XGBoost classification model.
4. Implement the Artificial Bee Colony algorithm from scratch using NumPy.
5. Apply ABC to jointly optimize feature selection and XGBoost hyperparameters.
6. Compare the optimized model against the baseline.
7. Evaluate performance using appropriate classification metrics.
8. Visualize ABC convergence, model performance, feature importance, and classification results.
9. Generate reproducible experimental results and a comprehensive academic report.

---

# 2. DATASET

## 2.1 Dataset Name

AI4I 2020 Predictive Maintenance Dataset.

Source: UCI Machine Learning Repository.

Official dataset page:

https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

Download the dataset from the official repository. The CSV file is named:

`ai4i2020.csv`

The dataset contains approximately 10,000 observations and 14 original columns.

The dataset is synthetic and was designed to represent realistic industrial predictive-maintenance scenarios. This limitation must be acknowledged in the final report.

## 2.2 Dataset Features

The dataset contains the following columns:

| Feature | Description | Type |
|---|---|---|
| UDI | Unique observation identifier | Integer |
| Product ID | Product identifier | Categorical |
| Type | Product quality category (L, M, H) | Categorical |
| Air temperature [K] | Ambient air temperature | Numerical |
| Process temperature [K] | Operating process temperature | Numerical |
| Rotational speed [rpm] | Machine rotational speed | Numerical |
| Torque [Nm] | Machine torque | Numerical |
| Tool wear [min] | Accumulated tool wear | Numerical |
| Machine failure | Target variable | Binary |
| TWF | Tool Wear Failure indicator | Failure-mode indicator |
| HDF | Heat Dissipation Failure indicator | Failure-mode indicator |
| PWF | Power Failure indicator | Failure-mode indicator |
| OSF | Overstrain Failure indicator | Failure-mode indicator |
| RNF | Random Failure indicator | Failure-mode indicator |

## 2.3 Target Variable

`Machine failure`

Binary classification:

- 0: Machine does not experience failure.
- 1: Machine experiences failure.

This is a highly imbalanced classification problem. The number of failure observations is much smaller than the number of non-failure observations.

The imbalance must be handled appropriately during training and evaluation.

## 2.4 Critical Data Leakage Prevention

This is an extremely important requirement.

The columns TWF, HDF, PWF, OSF, and RNF must NOT be used as input features.

These columns describe failure modes associated with the target and can directly reveal whether a machine failure occurred.

Using them as predictors would introduce target leakage and produce misleadingly high performance.

The following columns must also be removed:

- UDI: Unique observation identifier.
- Product ID: Identifier rather than a genuine operating measurement.

The `Type` feature must be retained because it represents a potentially useful machine/product category.

## 2.5 Final Input Features

The initial predictor set should contain:

1. Type
2. Air temperature [K]
3. Process temperature [K]
4. Rotational speed [rpm]
5. Torque [Nm]
6. Tool wear [min]

An additional engineered feature should be created:

7. Temperature difference = Process temperature − Air temperature.

The temperature difference may capture an aspect of the thermal operating condition.

The final model therefore has seven logical feature groups:

- Type
- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Temperature difference

The categorical Type feature should be one-hot encoded.

When performing feature selection, Type must be treated as one logical feature group. If selected, all its one-hot encoded columns must be included. Do not allow ABC to select only some of the dummy columns.

The temperature difference feature should be evaluated rather than assumed to improve performance.

---

# 3. DATA PREPROCESSING

Implement preprocessing in a dedicated module.

## 3.1 Data Cleaning

Perform the following:

- Load the CSV file using pandas.
- Inspect dataset shape.
- Check missing values.
- Check duplicate records.
- Check target distribution.
- Inspect data types.
- Remove identifier and failure-mode columns.
- Validate numerical feature ranges.
- Handle invalid or missing values if present.

Do not fabricate observations or artificially modify target labels.

## 3.2 Feature Engineering

Create:

`temperature_difference = process_temperature - air_temperature`

Preserve the original temperature features.

Compare the effect of using the engineered feature in the final feature selection experiment.

## 3.3 Encoding

Use OneHotEncoder or an equivalent pandas implementation for the categorical Type feature.

Ensure that the same encoded feature layout is used consistently throughout optimization and final evaluation.

## 3.4 Train, Validation and Test Split

Use a stratified split.

Recommended proportions:

- Training: 70%
- Validation: 15%
- Testing: 15%

Use a fixed random seed of 42 for the primary optimization experiment.

The split must preserve the class distribution.

The test set must remain completely isolated.

**The test set must never be used during:**

- ABC optimization.
- Hyperparameter selection.
- Feature selection.
- Early stopping.
- Threshold selection.
- Model selection.

The validation set is used for ABC fitness evaluation.

The test set is used only after the final model configuration has been selected.

## 3.5 Preprocessing Leakage Prevention

Any preprocessing operation that learns from data must be fitted only on the training portion.

This includes:

- Imputation, if required.
- Feature scaling, if used.
- Categorical encoding.

Apply the fitted transformations to validation and test data.

For tree-based XGBoost, numerical scaling is generally unnecessary. Do not add unnecessary scaling merely for the sake of preprocessing.

## 3.6 Class Imbalance

Inspect the class distribution before training.

Use XGBoost's `scale_pos_weight` parameter to account for class imbalance.

The default baseline value should be calculated from the training set:

`number_of_negative_samples / number_of_positive_samples`

The optimized model may tune this parameter within a reasonable bounded range.

Do not use SMOTE or oversampling on the validation or test set.

If additional imbalance-handling experiments are conducted, document them separately.

---

# 4. BASELINE MACHINE LEARNING MODEL

## 4.1 Selected Model

Use XGBoost Classifier.

Library:

`xgboost.XGBClassifier`

XGBoost is a gradient-boosted decision-tree algorithm that builds an ensemble of decision trees sequentially.

Each successive tree attempts to correct the errors made by the previous trees.

It is suitable for this project because:

- It performs well on structured/tabular data.
- It can model nonlinear relationships.
- It supports regularization.
- It supports class weighting.
- It supports early stopping.
- It has a relatively low computational cost for this dataset.
- Its hyperparameters provide a meaningful optimization search space.

## 4.2 Baseline Model

Build a baseline XGBoost classifier without ABC optimization.

Use the following initial configuration:

| Parameter | Value |
|---|---:|
| n_estimators | 300 |
| max_depth | 5 |
| learning_rate | 0.1 |
| min_child_weight | 1 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| gamma | 0 |
| reg_lambda | 1 |
| scale_pos_weight | Training class ratio |
| objective | binary:logistic |
| eval_metric | aucpr |
| tree_method | hist |
| random_state | 42 |

The baseline model must use all seven logical feature groups.

Train using the training data and use the validation data for early stopping.

Save:

- Baseline model.
- Baseline predictions.
- Baseline evaluation metrics.
- Training configuration.
- Training duration.

The baseline results will provide the reference against which the ABC-optimized model is evaluated.

---

# 5. ARTIFICIAL BEE COLONY (ABC) ALGORITHM

## 5.1 Introduction

Artificial Bee Colony is a population-based bio-inspired optimization algorithm inspired by the foraging behaviour of honeybees.

It was introduced by Dervis Karaboga.

The algorithm models how honeybees search for food sources and communicate information about their quality.

In this project:

- Each food source represents one candidate combination of features and XGBoost hyperparameters.
- The nectar quality represents the predictive performance of that candidate.
- Bees explore and exploit candidate solutions to identify a better configuration.

Implement the algorithm from scratch using NumPy.

Do not use ready-made ABC optimization libraries.

## 5.2 Main Components

ABC has three types of bees.

### A. Employed Bees

Each employed bee is associated with a food source.

It explores the neighbourhood of its current solution to find a potentially better candidate.

For each solution, generate a neighbouring solution using:

`v_ij = x_ij + phi_ij * (x_ij - x_kj)`

Where:

- `x_ij` is the current solution.
- `x_kj` is another randomly selected solution.
- `phi_ij` is a random number in [-1, 1].
- `v_ij` is the candidate neighbouring solution.

The candidate must be clipped or repaired to remain within the permitted search bounds.

If the candidate has better fitness, replace the current solution.

Otherwise, increment its trial counter.

### B. Onlooker Bees

Onlooker bees select promising food sources based on their fitness.

Calculate selection probabilities:

`p_i = fitness_i / sum(fitness)`

For a maximization objective, convert the objective value into a non-negative fitness measure.

Use roulette-wheel selection.

Selected food sources are explored further using the same neighbourhood generation strategy.

Better solutions replace inferior ones.

### C. Scout Bees

A food source that fails to improve for a specified number of trials is considered abandoned.

The corresponding employed bee becomes a scout.

It generates a new random candidate solution within the permitted search space.

Reset the trial counter.

This helps the algorithm escape local optima and maintain exploration.

## 5.3 ABC Algorithm Parameters

Use the following initial configuration:

| Parameter | Value |
|---|---:|
| Number of food sources | 10 |
| Employed bees | 10 |
| Onlooker bees | 10 |
| Maximum cycles | 10 |
| Abandonment limit | 5 |
| Maximum fitness evaluations | 240 |
| Random seed | 42 |

Count every actual model evaluation, including scout replacements.

The evaluation budget is a hard maximum, not a target that may be exceeded.

The final evaluation count may be lower if the configured stopping conditions are reached.

Record the exact evaluation count and number of optimization cycles completed.

The implementation should support changing these parameters through `config.yaml`.

## 5.4 Solution Representation

Each ABC candidate must encode:

- Seven feature-selection decisions.
- Eight XGBoost hyperparameters.

Use a normalized continuous vector:

`x ∈ [0,1]^15`

The first seven dimensions represent feature-selection decisions.

The remaining eight dimensions represent XGBoost hyperparameters.

### Feature-selection decoding

For each of the seven feature groups:

- Values greater than or equal to 0.5 mean selected.
- Values below 0.5 mean not selected.

If no features are selected, repair the candidate by selecting the feature with the highest corresponding continuous value.

This guarantees that the model always receives at least one feature.

Keep the Type feature as a single logical group, even though it expands into multiple encoded columns.

### Hyperparameter search space

The following hyperparameters must be optimized.

| Hyperparameter | Minimum | Maximum | Decoding |
|---|---:|---:|---|
| n_estimators | 100 | 600 | Integer |
| max_depth | 2 | 10 | Integer |
| learning_rate | 0.01 | 0.30 | Logarithmic |
| min_child_weight | 1 | 10 | Continuous |
| subsample | 0.60 | 1.00 | Continuous |
| colsample_bytree | 0.60 | 1.00 | Continuous |
| gamma | 0 | 5 | Continuous |
| scale_pos_weight | 1 | 30 | Continuous |

Use logarithmic decoding for learning_rate.

The other values should be decoded and clipped to their valid ranges.

Use deterministic decoding and record the actual decoded parameters for every evaluation.

The search-space bounds must be configurable.

---

# 6. FITNESS FUNCTION

## 6.1 Objective

The primary objective of ABC is to maximize validation Average Precision (AP), also called area under the precision-recall curve (PR-AUC in common reporting).

This metric is particularly relevant because the machine-failure class is rare.

Accuracy alone must not be used as the optimization objective.

## 6.2 Fitness Evaluation Procedure

For every candidate solution:

1. Decode the feature mask.
2. Decode the XGBoost hyperparameters.
3. Select the corresponding feature groups.
4. Train an XGBoost classifier on the training data.
5. Evaluate it on the validation data.
6. Calculate Average Precision using predicted probabilities.
7. Return the validation AP as the objective score.
8. Record all parameters, metrics and training duration.
9. Clear temporary objects and release resources where appropriate.

Use XGBoost's CPU histogram tree method.

Use early stopping where supported by the installed XGBoost version.

If early stopping is used, make the implementation compatible with the installed API and document the effective number of boosting rounds.

## 6.3 Objective Direction

The ABC algorithm must maximize validation AP.

A larger AP value indicates better performance.

For roulette-wheel selection, convert the AP objective into a positive fitness value that remains valid even when objective scores are close to zero.

Use a consistent numerical strategy and unit-test it.

## 6.4 Caching

Implement fitness caching.

The cache key must contain:

- Decoded feature mask.
- Decoded hyperparameters.
- Any evaluation settings that materially change the result.

Candidates that decode to the same configuration should reuse their existing evaluation.

Log cache hits separately.

A cache hit must not be represented as a new model training run.

Track:

- Total optimizer requests.
- Actual model evaluations.
- Cache hits.
- Remaining evaluation budget.

Persist the cache so it survives a Colab disconnection.

---

# 7. EXPERIMENTAL METHODOLOGY

The experiments must be conducted in the following order.

## Experiment 1: Exploratory Data Analysis

Analyze:

- Dataset dimensions.
- Data types.
- Missing values.
- Duplicate observations.
- Target class distribution.
- Numerical feature distributions.
- Feature correlations.
- Failure frequency across product types.
- Failure patterns against temperature.
- Failure patterns against rotational speed.
- Failure patterns against torque.
- Failure patterns against tool wear.

Generate appropriate plots.

## Experiment 2: Baseline XGBoost

Train the baseline model.

Record:

- Training time.
- Validation AP.
- Validation precision.
- Validation recall.
- Validation F1-score.
- Test metrics after final evaluation.

## Experiment 3: ABC Optimization

Execute the custom ABC algorithm.

At each candidate evaluation, record:

- Evaluation ID.
- Optimization cycle.
- Bee type.
- Candidate vector.
- Decoded feature mask.
- Decoded hyperparameters.
- Validation AP.
- Cache status.
- Training duration.
- Best-so-far AP.

Maintain the global best solution.

At the end of optimization, save the best feature mask and hyperparameters.

## Experiment 4: Final ABC-XGBoost Model

Retrain the best ABC configuration.

For the final model, combine the original training and validation sets after all configuration choices are frozen.

Fit preprocessing on this combined development dataset.

Train the final model using the selected configuration.

The test set must remain untouched until final evaluation.

For a fair comparison, retrain the baseline XGBoost using the same combined development dataset.

Evaluate both models on exactly the same test observations.

## Experiment 5: Multiple Random Seeds

Perform final evaluation across five seeds:

`[0, 1, 2, 3, 4]`

Use the same selected ABC configuration for the five final evaluation runs so that the comparison measures training variability rather than repeatedly changing the optimization configuration.

For each seed:

- Train baseline XGBoost.
- Train ABC-optimized XGBoost.
- Generate test probabilities.
- Calculate evaluation metrics.

Report mean and standard deviation.

Also perform three independent ABC search runs with different random seeds, if runtime permits, to assess optimization stability.

Do not select configurations based on test performance.

---

# 8. EVALUATION METRICS

Because the dataset is imbalanced, use several complementary evaluation metrics.

## 8.1 Accuracy

Measures the proportion of correctly classified observations.

Report it, but do not rely on it alone.

## 8.2 Precision

Measures the proportion of predicted failures that are actual failures.

## 8.3 Recall

Measures the proportion of actual failures correctly identified.

This is particularly important in predictive maintenance because missed failures can lead to equipment downtime.

## 8.4 F1-score

Harmonic mean of precision and recall.

Report:

- Positive-class F1.
- Macro F1.
- Weighted F1.

## 8.5 ROC-AUC

Measures the model's ability to distinguish between failure and non-failure observations across classification thresholds.

## 8.6 Average Precision / PR-AUC

The primary optimization metric.

Use scikit-learn's `average_precision_score` consistently.

Clearly distinguish Average Precision from trapezoidal integration of the precision-recall curve if both are reported.

## 8.7 Confusion Matrix

Report:

- True Positives.
- True Negatives.
- False Positives.
- False Negatives.

Generate a confusion matrix for both baseline and optimized models.

## 8.8 Threshold Handling

Use a fixed threshold of 0.5 for the primary baseline-versus-optimized comparison.

Optionally perform a separate threshold-selection experiment using validation data.

If an alternative threshold is chosen, apply it unchanged to the test set and report it explicitly.

Never select a threshold using test labels.

---

# 9. REQUIRED VISUALIZATIONS

Generate publication-quality figures with proper titles, labels, legends and readable formatting.

Save all figures as PNG files at 150–200 DPI.

Required plots:

### Figure 1: Target Class Distribution

Show the imbalance between failure and non-failure observations.

### Figure 2: Feature Correlation Heatmap

Show relationships between numerical operating features.

### Figure 3: Feature Distributions

Show histograms or boxplots of the numerical features.

### Figure 4: Failure Distribution by Product Type

Compare failure rates across L, M and H product types.

### Figure 5: ABC Convergence Curve

Plot best-so-far validation AP against optimization cycle.

Also show the population's mean AP.

### Figure 6: ABC Best Fitness Against Evaluation Count

Plot the best-so-far validation AP against the number of actual fitness evaluations.

### Figure 7: ABC Exploration and Exploitation

Visualize the number of successful improvements and scout replacements per cycle.

### Figure 8: Baseline vs ABC-XGBoost Metric Comparison

Compare accuracy, precision, recall, F1, ROC-AUC and AP.

### Figure 9: Baseline Confusion Matrix

Visualize the baseline test results.

### Figure 10: Optimized Confusion Matrix

Visualize the ABC-XGBoost test results.

### Figure 11: ROC Curves

Plot baseline and optimized ROC curves together.

### Figure 12: Precision-Recall Curves

Plot baseline and optimized precision-recall curves together.

### Figure 13: Feature Importance

Show the importance of features selected by ABC.

### Figure 14: Feature Selection Frequency

If multiple ABC runs are performed, show how frequently each feature group is selected in the best solutions.

### Figure 15: Optimization Search Analysis

Visualize the distribution of candidate AP scores and the progression of best configurations.

All visualizations must be generated from actual experiment outputs.

Never hard-code fabricated values.

---

# 10. REQUIRED TABLES

Save tables in CSV and Markdown formats.

## Table 1: Dataset Summary

Include feature names, descriptions, types and usage decisions.

## Table 2: Preprocessing Summary

Document:

- Dropped columns.
- Retained columns.
- Engineered features.
- Encoding.
- Train/validation/test sizes.
- Class distribution.

## Table 3: Baseline Results

Include all evaluation metrics.

## Table 4: ABC Optimization Parameters

Document population size, employed bees, onlooker bees, abandonment limit, cycle count and evaluation budget.

## Table 5: ABC Best Configuration

Include:

- Selected features.
- n_estimators.
- max_depth.
- learning_rate.
- min_child_weight.
- subsample.
- colsample_bytree.
- gamma.
- scale_pos_weight.
- Validation AP.

## Table 6: Optimization History

Include the best-so-far and population mean fitness per cycle.

## Table 7: Final Model Comparison

Compare:

- Baseline XGBoost.
- ABC-XGBoost.

Report mean ± standard deviation over five seeds.

## Table 8: Runtime Comparison

Include:

- Preprocessing time.
- Baseline training time.
- ABC search time.
- Final training time.
- Total experiment time.
- Actual fitness evaluation count.
- Cache hit count.

---

# 11. PROJECT DIRECTORY STRUCTURE

Build a clean, modular project with the following structure:

```text
abc_machine_failure/
│
├── config.yaml
├── requirements.txt
├── README.md
├── PROJECT_CONTEXT.md
│
├── data/
│   ├── raw/
│   │   └── ai4i2020.csv
│   │
│   └── processed/
│
├── src/
│   ├── __init__.py
│   │
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── eda.py
│   ├── feature_engineering.py
│   │
│   ├── baseline.py
│   ├── model.py
│   ├── search_space.py
│   │
│   ├── abc_optimizer.py
│   ├── fitness.py
│   ├── evaluation.py
│   │
│   ├── metrics.py
│   ├── visualization.py
│   ├── experiment.py
│   └── utils.py
│
├── results/
│   ├── logs/
│   ├── tables/
│   ├── figures/
│   ├── models/
│   ├── predictions/
│   └── summary.md
│
├── checkpoints/
│   ├── abc_state.json
│   ├── fitness_cache.json
│   └── optimization_state.json
│
└── run_experiments.py
```

All source code must be modular and documented.

Use:

- Type hints.
- Docstrings.
- Meaningful variable names.
- Proper exception handling.
- Centralized configuration.
- Reproducible random seeds.

Avoid hard-coded parameters outside `config.yaml`.

---

# 12. CONFIGURATION MANAGEMENT

All important parameters must be defined in `config.yaml`.

The configuration should include:

### Dataset

- Dataset path.
- Target column.
- Identifier columns.
- Leakage columns.
- Feature columns.
- Test size.
- Validation size.
- Random seed.

### XGBoost

- Default baseline hyperparameters.
- Search-space bounds.
- Early stopping settings.
- CPU tree method.
- Number of final seeds.

### ABC

- Food source count.
- Employed bee count.
- Onlooker bee count.
- Maximum cycles.
- Abandonment limit.
- Evaluation budget.
- Random seed.

### Execution

- Output directories.
- Logging configuration.
- Checkpoint frequency.
- Resume option.
- Number of independent optimizer runs.

### Visualization

- Figure DPI.
- Figure dimensions.
- Output format.
- Plot style.

---

# 13. CHECKPOINTING AND RESUME SUPPORT

This is mandatory.

The experiment must survive interruptions, especially when executed in Google Colab.

Implement:

1. Save the ABC population after every completed optimization cycle.
2. Save the best solution.
3. Save all food-source fitness values.
4. Save trial counters.
5. Save the current cycle number.
6. Save the random generator state.
7. Save the fitness cache.
8. Save the evaluation log after every actual model evaluation.
9. Resume from the last successfully completed checkpoint.

Avoid losing completed model evaluations if the session disconnects.

Use atomic file writes where practical to avoid corrupting checkpoints.

Do not restart the entire search when valid checkpoints exist.

---

# 14. COMMAND-LINE INTERFACE

Create a single entry point:

`run_experiments.py`

Support these commands:

### Exploratory data analysis

```bash
python run_experiments.py --stage eda
```

### Baseline training

```bash
python run_experiments.py --stage baseline
```

### Quick ABC test

```bash
python run_experiments.py --stage search --quick
```

### Full ABC optimization

```bash
python run_experiments.py --stage search
```

### Final model evaluation

```bash
python run_experiments.py --stage final
```

### Generate plots and tables

```bash
python run_experiments.py --stage plots
```

### Execute complete pipeline

```bash
python run_experiments.py --stage all
```

### Resume optimization

```bash
python run_experiments.py --stage search --resume
```

The CLI may be extended with options for selecting an optimizer run, changing seeds or overriding configuration values.

## Quick Mode

The quick mode should be a smoke test.

Suggested configuration:

- Food sources: 4.
- Employed bees: 4.
- Onlooker bees: 4.
- Maximum cycles: 2.
- Small evaluation budget.
- Reduced XGBoost estimator limit.

It must exercise:

- Dataset loading.
- Preprocessing.
- Baseline model.
- ABC initialization.
- Employed bee phase.
- Onlooker bee phase.
- Scout logic.
- Fitness evaluation.
- Caching.
- Logging.
- Checkpointing.
- Final metric generation.

The quick mode is for debugging, not for final academic results.

---

# 15. IMPLEMENTATION ORDER

Do not attempt to build the entire system in one step.

Implement and verify incrementally.

### Phase 1: Dataset and preprocessing

1. Download and load AI4I 2020.
2. Validate all columns.
3. Inspect class imbalance.
4. Remove identifiers and leakage columns.
5. Engineer temperature difference.
6. Implement categorical encoding.
7. Implement stratified splitting.
8. Verify there is no preprocessing leakage.

Expected output: Clean training, validation and test datasets.

### Phase 2: Baseline model

1. Implement XGBoost classifier.
2. Train the baseline.
3. Calculate validation metrics.
4. Generate baseline confusion matrix.
5. Save model and predictions.

Expected output: Working baseline and reproducible metrics.

### Phase 3: ABC implementation

1. Implement candidate encoding and decoding.
2. Implement feature-mask decoding.
3. Implement hyperparameter decoding.
4. Implement employed bees.
5. Implement onlooker bees.
6. Implement roulette-wheel selection.
7. Implement scout bees.
8. Implement abandonment logic.
9. Implement boundary repair.
10. Implement best-solution tracking.

Expected output: A working ABC optimizer.

### Phase 4: Fitness evaluation

1. Connect ABC to XGBoost.
2. Implement validation AP calculation.
3. Implement caching.
4. Implement evaluation logging.
5. Implement checkpointing.

Expected output: End-to-end optimization.

### Phase 5: Verification

Test:

- ABC population initialization.
- Search bounds.
- Feature selection.
- Hyperparameter decoding.
- Fitness direction.
- Roulette selection.
- Trial counters.
- Scout replacement.
- Caching.
- Checkpoint recovery.
- Reproducibility.

### Phase 6: Full optimization

Run ABC with the full configuration.

Save the best candidate.

### Phase 7: Final experiments

1. Retrain baseline.
2. Retrain optimized model.
3. Evaluate over five seeds.
4. Generate predictions.
5. Compute mean and standard deviation.

### Phase 8: Reporting

Generate all required:

- CSV tables.
- Markdown tables.
- Figures.
- Model files.
- Optimization logs.
- Runtime summaries.
- Final summary report.

---

# 16. TESTING AND CORRECTNESS REQUIREMENTS

The following conditions must be verified.

1. The target is never included in the feature matrix.
2. UDI and Product ID are not used as predictors.
3. TWF, HDF, PWF, OSF and RNF are excluded.
4. Preprocessing is fitted only on the appropriate training data.
5. The validation set is used only for model selection and optimization.
6. The test set is used only for final evaluation.
7. The feature mask always contains at least one selected feature group.
8. Hyperparameters always remain within configured bounds.
9. Every ABC candidate is evaluated using the same split and evaluation procedure.
10. Cache keys correctly represent decoded configurations.
11. Cache hits do not trigger redundant training.
12. Scout replacement logic is correctly implemented.
13. The optimizer never exceeds its evaluation budget.
14. Checkpoint recovery produces a valid continuation.
15. Results are reproducible within the limitations of the software and CPU environment.

Create unit tests for the ABC operators and fitness function.

---

# 17. PERFORMANCE EXPECTATIONS AND RUNTIME

The project is designed to run on CPU.

Recommended environment:

- Python 3.10 or newer.
- NumPy.
- pandas.
- scikit-learn.
- XGBoost.
- Matplotlib.
- Seaborn.
- PyYAML.
- tqdm.

No CUDA installation is necessary.

Use XGBoost's `tree_method="hist"` and CPU execution.

Expected runtime is hardware-dependent.

Initial planning estimate:

| Stage | Estimated time |
|---|---:|
| Data loading and preprocessing | 1–3 minutes |
| EDA | 1–3 minutes |
| Baseline model | 1–3 minutes |
| ABC optimization | 5–25 minutes |
| Final model evaluation | 2–8 minutes |
| Visualizations and reporting | 1–3 minutes |

These are approximate estimates, not guaranteed execution times. Measure actual runtimes and report them.

If optimization is slower than expected, profile the fitness evaluation before changing the experimental design.

Do not silently reduce the evaluation budget or change the search space in the final experiment.

---

# 18. EXPECTED OUTPUTS

The completed project must generate:

### Models

- `baseline_xgb.json`
- `abc_optimized_xgb.json`

### Optimization results

- `abc_evaluations.csv`
- `abc_convergence.csv`
- `abc_best_config.json`
- `abc_feature_mask.json`
- `abc_summary.json`

### Evaluation tables

- `dataset_summary.csv`
- `preprocessing_summary.csv`
- `baseline_results.csv`
- `optimized_results.csv`
- `master_comparison.csv`
- `runtime_comparison.csv`

### Visualizations

All 15 required figures listed in Section 9.

### Predictions

- `baseline_test_predictions.csv`
- `optimized_test_predictions.csv`

### Final report

`results/summary.md`

The summary must contain:

1. Project overview.
2. Dataset information.
3. Preprocessing decisions.
4. Baseline model configuration.
5. ABC algorithm configuration.
6. Best hyperparameters.
7. Selected features.
8. Optimization convergence.
9. Final performance comparison.
10. Runtime comparison.
11. Important observations.
12. Limitations.
13. Any deviations from the original specification.

All observations must be derived from actual experiment results.

---

# 19. EXPECTED ACADEMIC DISCUSSION

The final report should answer these questions:

1. Why is XGBoost appropriate for predictive maintenance?
2. Why is the AI4I dataset suitable for this problem?
3. Why is the dataset imbalanced?
4. Why is accuracy insufficient as the only metric?
5. How does ABC explore the hyperparameter search space?
6. What are the roles of employed, onlooker and scout bees?
7. How does feature selection influence the model?
8. Which features were selected by ABC?
9. Which hyperparameters were most influential?
10. Did ABC improve validation AP?
11. Did the improvement generalize to the test set?
12. How does ABC compare with the baseline?
13. What is the computational cost of optimization?
14. What are the limitations of using a synthetic dataset?
15. How could the system be improved in future work?

Discuss both successful and unsuccessful results honestly.

Do not claim that ABC improves the model unless the actual experimental results support that conclusion.

---

# 20. IMPORTANT IMPLEMENTATION INSTRUCTIONS FOR THE CODING AGENT

You are acting as a senior machine learning engineer and research implementation assistant.

Your responsibility is to build a complete, reproducible and academically defensible implementation of this project.

Follow these instructions strictly:

1. Read this entire specification before writing code.
2. Do not immediately generate the entire codebase in one response.
3. First propose the implementation plan and confirm the project structure.
4. Implement the system incrementally.
5. Use actual dataset observations. Never fabricate data or experimental results.
6. Implement ABC from scratch using NumPy.
7. Do not use prebuilt ABC or metaheuristic optimization libraries.
8. Use XGBoost for the predictive model.
9. Do not use a GPU unless explicitly requested.
10. Do not introduce unnecessary deep-learning frameworks.
11. Do not use the test set during optimization.
12. Prevent target leakage rigorously.
13. Do not optimize accuracy at the expense of minority-class performance.
14. Save all experimental outputs.
15. Implement checkpointing and recovery.
16. Keep all configurable parameters in `config.yaml`.
17. Add docstrings and type hints to every function.
18. Write unit tests for important algorithm components.
19. Ensure that all generated figures and tables use actual results.
20. Document any deviations from this specification.

## Final execution philosophy

Prioritize correctness, reproducibility, runtime efficiency and academic quality.

The objective is not simply to achieve the highest possible accuracy.

The objective is to demonstrate a complete bio-inspired optimization workflow, with a clear baseline, a properly implemented ABC algorithm, meaningful optimization, fair evaluation and defensible conclusions.

**Do not sacrifice experimental correctness to obtain artificially impressive results.**

The completed implementation must be suitable for a 30-mark Bio-Inspired Algorithms academic assessment at B.M.S. College of Engineering.
