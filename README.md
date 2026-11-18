<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Ensemble-Learning-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Signal-Processing-blueviolet?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Feature-Engineering-teal?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<h1 align="center">Exoplanet Classification Using Ensemble Learning</h1>

<p align="center">
  Advanced Machine Learning Framework for Exoplanet Classification
  Using Statistical Feature Engineering and Ensemble Models
</p>

<p align="center">
  Signal Feature Extraction • Correlation-Based Feature Reduction •
  Ensemble Classification • Model Evaluation
</p>

---

## Overview

**Exoplanet Classification Using Ensemble Learning** is a machine learning project focused on classifying astronomical observations using engineered statistical and signal-derived features.

The project implements a structured classification workflow that transforms high-dimensional input measurements into a compact feature representation. It combines statistical descriptors, frequency-domain characteristics, peak-based measurements, and signal-difference features to represent patterns in the input data.

The workflow incorporates correlation-based feature reduction, feature scaling, comparative model evaluation, and ensemble learning. A trained Random Forest classifier is persisted alongside preprocessing artifacts and metadata to support reproducible model usage.

The repository demonstrates an end-to-end applied machine learning workflow covering:

- Data preprocessing and feature preparation
- Statistical feature engineering
- Fourier Transform-based feature extraction
- Peak-based and signal-difference analysis
- Correlation-based feature reduction
- Feature scaling and column management
- Comparative evaluation of classification algorithms
- Voting and stacking ensemble experiments
- Classification threshold analysis
- Feature-group ablation experiments
- Noise robustness evaluation
- Model persistence and experiment result export

The notebook is the primary implementation and experimentation interface. The repository also contains serialized model artifacts and exported evaluation results.

---

## Why This Project Matters

Machine learning can help analyze complex astronomical observations by identifying statistical patterns and relationships within measured signals.

Classification of astronomical time-series data presents several challenges, including high-dimensional inputs, potentially redundant features, subtle signal characteristics, and the need to evaluate model performance beyond overall accuracy.

This project explores how engineered signal representations, feature reduction, and ensemble learning can be combined in a conventional machine learning pipeline.

The objective is to investigate the effectiveness of different classification algorithms and feature representations for exoplanet-related classification tasks.

### Potential Applications

- Astronomical time-series classification
- Exoplanet detection research
- Signal characterization and analysis
- Machine learning-assisted candidate screening
- Feature engineering for scientific datasets
- Comparative evaluation of classification algorithms
- Research in computational astrophysics

Model predictions represent statistical classifications learned from the training data. They do not independently establish the existence of an exoplanet or replace scientific validation.

---

## Key Technical Highlights

- End-to-end supervised classification workflow
- Statistical feature extraction from high-dimensional input signals
- Frequency-domain analysis using Fourier Transform features
- Peak-based signal characterization
- Consecutive-observation difference analysis
- Correlation-based removal of redundant features
- Standardized feature transformations
- Random Forest classifier with persisted model artifacts
- Comparative evaluation of multiple classification algorithms
- XGBoost, CatBoost, and LightGBM model experiments
- Logistic Regression baseline
- Soft voting, weighted voting, and stacking ensemble experiments
- Classification probability threshold configuration
- Feature-group ablation analysis
- Noise robustness experiments
- Exported model metrics and pipeline metadata

---

## Research Contributions

- Developed statistical and signal-derived features to represent input observations.
- Combined distribution statistics with frequency-domain, peak-based, and difference-based descriptors.
- Applied correlation-based reduction to remove redundant engineered features.
- Compared individual classifiers and ensemble configurations using multiple evaluation metrics.
- Examined the influence of different feature groups through ablation experiments.
- Evaluated recorded model performance under different injected noise levels.
- Persisted the selected classifier, scaling transformation, feature metadata, and classification threshold.

---

## Core Machine Learning Components

| Component | Purpose |
|---|---|
| Statistical Feature Engineering | Captures distribution and variability characteristics |
| Fourier Transform Analysis | Extracts frequency-domain information |
| Peak-Based Features | Characterizes detected peaks in the input signal |
| Difference-Based Features | Represents changes between consecutive measurements |
| Correlation Analysis | Identifies and removes redundant features |
| StandardScaler | Standardizes selected feature values |
| Random Forest | Persisted classification model |
| XGBoost | Gradient-boosted tree classification |
| CatBoost | Gradient boosting classification |
| LightGBM | Gradient boosting classification |
| Logistic Regression | Linear classification baseline |
| Voting Ensembles | Combines predictions from multiple classifiers |
| Stacking Ensemble | Combines base-model predictions through a meta-estimator |
| Classification Threshold | Converts predicted probabilities into class decisions |

---

## Repository Structure

```text
Exoplanet-Classification-Using-Ensemble-Learning/
│
├── Advanced Exoplanet Classification using Feature Engineering
│   and Ensemble Models.ipynb.ipynb
│
├── catboost_info/
│   ├── learn/
│   ├── catboost_training.json
│   ├── learn_error.tsv
│   └── time_left.tsv
│
├── rf_exoplanet_model.pkl
├── feature_scaler.pkl
├── raw_feature_columns.pkl
├── selected_feature_columns.pkl
├── dropped_correlated_features.pkl
│
├── best_threshold.txt
├── pipeline_info.json
│
├── model_results.csv
├── ablation_results.csv
├── noise_results.csv
│
└── README.md
```

### Repository Artifacts

| File | Description |
|---|---|
| `Advanced Exoplanet Classification using Feature Engineering and Ensemble Models.ipynb.ipynb` | Main notebook containing data processing, feature engineering, model experiments, and evaluation |
| `rf_exoplanet_model.pkl` | Serialized Random Forest classifier |
| `feature_scaler.pkl` | Persisted feature scaling transformation |
| `raw_feature_columns.pkl` | Stored raw input feature schema |
| `selected_feature_columns.pkl` | Names of selected engineered features |
| `dropped_correlated_features.pkl` | Features recorded as removed during correlation-based reduction |
| `best_threshold.txt` | Stored classification threshold |
| `pipeline_info.json` | Model metadata, recorded ROC-AUC, threshold, selected features, removed features, and data dimensions |
| `model_results.csv` | Comparative classification results |
| `ablation_results.csv` | Evaluation results for different feature groups |
| `noise_results.csv` | Recorded noise robustness experiment results |
| `catboost_info/` | CatBoost training logs and metadata |

The notebook is the primary training and experimentation entry point. The repository does not currently include a separate inference script or web application.

---

## Dataset and Input Representation

The repository contains the notebook, trained model artifacts, feature metadata, and experiment results. The original input dataset files are not included in the repository listing.

The dataset's exact provenance, external source, and original class-label mapping should be confirmed from the notebook's data-loading and preprocessing cells before being documented.

### Recorded Data Dimensions

The committed `pipeline_info.json` contains the following dimensions:

| Dataset Partition | Recorded Shape |
|---|---:|
| Training data | 5,087 × 3,197 |
| Test data | 570 × 3,197 |

These values are taken from the saved pipeline metadata. They describe the recorded training and test input dimensions; they do not independently establish the dataset's source or class distribution.

### Input Representation

The saved feature metadata indicates that the model workflow extracts a compact set of engineered features from a larger input representation.

The selected feature names include statistical descriptors, frequency-domain features, peak characteristics, and signal-difference measurements.

Before reproducing the training workflow, inspect the notebook's data-loading cells to identify the required input files, expected paths, preprocessing steps, and original target-label definitions.

---

## System Architecture

### End-to-End Pipeline Overview

1. Load the input dataset and inspect its structure.
2. Separate classification targets from input measurements.
3. Perform exploratory analysis and inspect the class distribution.
4. Extract statistical features from the input measurements.
5. Calculate frequency-domain features using Fourier analysis.
6. Generate peak-based and signal-difference features.
7. Identify and remove correlated engineered features.
8. Preserve consistent feature names and column ordering.
9. Apply the feature scaling transformation.
10. Train and evaluate individual classification models.
11. Compare voting and stacking ensemble configurations.
12. Evaluate classification metrics and decision thresholds.
13. Perform feature-group ablation experiments.
14. Evaluate recorded performance under injected noise.
15. Persist model artifacts, metadata, and evaluation results.

---

## Feature Engineering Methodology

The project uses engineered features to represent different characteristics of the input signal.

### Statistical Features

Statistical descriptors summarize the distribution and variability of the measurements.

| Feature | Description |
|---|---|
| `mean` | Mean of the input measurements |
| `std` | Standard deviation |
| `min` | Minimum value |
| `max` | Maximum value |
| `median` | Median value |
| `skew` | Distribution asymmetry |
| `kurtosis` | Distribution tail characteristics |
| `q25` | First quartile |
| `q75` | Third quartile |
| `energy` | Aggregate signal energy |

### Frequency-Domain Features

Fourier Transform-based analysis produces features that characterize the frequency composition of the input signal.

| Feature | Description |
|---|---|
| `fft_mean` | Mean of the computed frequency-domain representation |
| `fft_peak_freq` | Frequency associated with the dominant detected spectral peak |

### Peak-Based Features

Peak-based descriptors characterize prominent structures detected in the signal.

| Feature | Description |
|---|---|
| `num_peaks` | Number of detected peaks |
| `peak_mean` | Mean value of detected peaks |
| `peak_min` | Minimum value among detected peaks |

### Signal-Difference Features

Difference-based features describe variation between consecutive measurements.

| Feature | Description |
|---|---|
| `mean_abs_diff` | Mean absolute difference between consecutive observations |
| `diff_energy` | Aggregate energy of the difference signal |
| `zero_crossings` | Number of zero crossings according to the implemented feature definition |

These feature groups provide complementary representations of distribution, frequency, peak, and variation characteristics.

The exact calculations and parameter choices are defined in the notebook.

---

## Feature Selection and Preprocessing

### Correlation-Based Feature Reduction

The project records features removed during correlation-based reduction in `dropped_correlated_features.pkl`.

The persisted metadata identifies the following features as removed:

```text
range
iqr
mad
rms
fft_std
fft_max
fft_p90
fft_p95
fft_p99
peak_std
peak_max
std_diff
max_abs_diff
```

Removing redundant features can reduce feature dimensionality and limit duplication of similar information.

The notebook remains the reference for the exact correlation calculation, threshold, and removal procedure.

### Selected Features

The committed `pipeline_info.json` records the following 18 selected features:

```text
mean
std
min
max
median
skew
kurtosis
q25
q75
energy
zero_crossings
fft_mean
fft_peak_freq
num_peaks
peak_mean
peak_min
mean_abs_diff
diff_energy
```

These features capture statistical properties, frequency characteristics, peak information, and signal variation.

### Feature Scaling

The repository contains `feature_scaler.pkl`, which stores a fitted feature scaling transformation.

For compatible inference, the same feature definitions, selected columns, column ordering, and scaling procedure used during training must be preserved.

The saved metadata and feature artifacts should be used together with the classifier rather than treating the model file as a complete preprocessing pipeline.

---

## Classification Models

The recorded comparative results cover the following individual models and ensemble configurations.

| Model | Role |
|---|---|
| Random Forest | Tree-based classifier and persisted model |
| XGBoost | Gradient-boosted tree classifier |
| CatBoost | Gradient boosting classifier |
| LightGBM | Gradient boosting classifier |
| Logistic Regression | Linear classification baseline |
| Soft Voting Ensemble | Combines model probability estimates |
| Weighted Voting Ensemble | Combines predictions using weighted contributions |
| Stacking Ensemble | Combines base-model predictions using a meta-estimator |

The committed `pipeline_info.json` identifies Random Forest as the selected model and records a ROC-AUC of approximately `0.9949`.

The repository contains a serialized Random Forest model. Separate serialized artifacts for all the other evaluated classifiers are not listed in the repository.

---

## Evaluation Framework

The project records multiple classification metrics to assess model performance.

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of observations classified correctly |
| Precision | Reliability of positive predictions |
| Recall | Proportion of actual positive observations detected |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Measures class discrimination across probability thresholds |

Accuracy alone can be misleading when class distributions are imbalanced or when missing positive cases has a significant impact.

Precision, recall, F1 score, and ROC-AUC should therefore be considered together when comparing classification configurations.

---

## Results Snapshot

The following values are recorded in the repository's `model_results.csv`. The listed precision, recall, F1, and accuracy values correspond to the threshold recorded in that file.

| Model | ROC-AUC | Accuracy | Precision | Recall | F1 Score |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.9949 | 0.9947 | 1.0000 | 0.4000 | 0.5714 |
| Weighted Ensemble (RF+XGB+CAT) | 0.9940 | 0.9965 | 1.0000 | 0.6000 | 0.7500 |
| Stacking Ensemble | 0.9876 | 0.9965 | 1.0000 | 0.6000 | 0.7500 |
| XGBoost | 0.9756 | 0.9965 | 1.0000 | 0.6000 | 0.7500 |
| Soft Voting Ensemble | 0.9614 | 0.9965 | 1.0000 | 0.6000 | 0.7500 |
| CatBoost | 0.9349 | 0.9965 | 1.0000 | 0.6000 | 0.7500 |
| Logistic Regression | 0.8945 | 0.6860 | 0.0272 | 1.0000 | 0.0529 |
| LightGBM | 0.8931 | 0.9912 | 0.0000 | 0.0000 | 0.0000 |

These are recorded experiment results and have not been independently reproduced for this documentation.

### Results Interpretation

- Random Forest has the highest recorded ROC-AUC among the models listed.
- The weighted ensemble records higher accuracy and F1 score than Random Forest at the threshold used in the comparison.
- The recorded Random Forest recall of `0.4000` indicates that its classification results warrant closer examination of missed positive observations.
- Logistic Regression and LightGBM demonstrate why accuracy should not be interpreted in isolation.
- The model-selection decision should account for the intended classification objective, the class-label mapping, and the relative costs of false positives and false negatives.

The results should be validated using the notebook's evaluation procedure before being treated as independently reproduced performance.

---

## Classification Threshold

The repository contains a `best_threshold.txt` artifact with the following value:

```text
0.16
```

The saved `pipeline_info.json` also records a threshold of `0.16`.

A classification threshold determines how predicted probabilities are converted into class decisions. Adjusting the threshold changes the trade-off between positive-class recall and precision.

A lower threshold can increase positive predictions, potentially improving recall while also increasing false positives.

### Threshold Consistency

The committed `model_results.csv` records a threshold of `0.5` for the listed model comparisons, while the saved threshold artifact records `0.16`.

These values represent different recorded configurations. Re-evaluate the selected classifier using the intended threshold before reporting threshold-specific performance. For reliable evaluation, threshold selection should be performed using validation data rather than optimized directly on the final test set.

---

## Ablation Analysis

The repository contains `ablation_results.csv`, which records classification performance for different feature groups.

| Feature Set | ROC-AUC | Accuracy | F1 Score |
|---|---:|---:|---:|
| All features | 0.9949 | 0.9947 | 0.5714 |
| Statistical + FFT + Peak + Complexity | 0.9949 | 0.9947 | 0.5714 |
| Statistical only | 0.9912 | 0.9947 | 0.5714 |
| Statistical + FFT + Peak | 0.9885 | 0.9947 | 0.5714 |
| Statistical + FFT | 0.9821 | 0.9947 | 0.5714 |

The recorded experiments compare the predictive performance of different engineered feature groups.

The results indicate that statistical features provide substantial discrimination in the evaluated configuration. Adding other feature groups changes ROC-AUC, while the listed accuracy and F1 values remain unchanged.

These are results from the committed experiment output and should be interpreted within the context of its evaluation procedure.

---

## Noise Robustness Analysis

The repository includes `noise_results.csv`, which records ROC-AUC values for different noise levels.

| Noise Level | Recorded ROC-AUC |
|---|---:|
| 0.01 | 0.9313 |
| 0.05 | 0.9499 |
| 0.10 | 0.9411 |
| 0.20 | 0.7172 |

The recorded results show a decrease in ROC-AUC at the highest listed noise level.

The values at lower noise levels are not strictly monotonic. Repeated experiments with controlled random seeds and documented perturbation procedures would help establish a more reliable assessment of noise sensitivity.

---

## Model Artifacts

The repository contains the following persisted model and preprocessing components.

| Artifact | Description |
|---|---|
| `rf_exoplanet_model.pkl` | Trained Random Forest classifier |
| `feature_scaler.pkl` | Fitted feature scaling transformation |
| `raw_feature_columns.pkl` | Expected raw feature schema |
| `selected_feature_columns.pkl` | Selected engineered feature names |
| `dropped_correlated_features.pkl` | Feature names recorded as removed |
| `best_threshold.txt` | Stored classification threshold |
| `pipeline_info.json` | Model selection, recorded ROC-AUC, threshold, feature metadata, and data dimensions |

For compatible inference, new observations must be transformed using the same feature engineering, feature selection, column alignment, and scaling sequence used during training.

The repository does not currently provide a separate inference script that wraps all of these steps into a single callable interface.

**Security note:** Pickle files can execute code when deserialized. Only load artifacts from trusted sources. Compatibility between the library versions used during training and inference should also be verified.

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantkothari/Exoplanet-Classification-Using-Ensemble-Learning.git

cd Exoplanet-Classification-Using-Ensemble-Learning
```

### Create a Virtual Environment

**Windows (PowerShell / Command Prompt)**

```bash
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux**

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

The repository listing does not currently include a `requirements.txt` file.

Install the principal libraries used by the documented modeling workflow:

```bash
python -m pip install --upgrade pip

pip install numpy pandas scipy scikit-learn matplotlib seaborn jupyter xgboost lightgbm catboost
```

The exact dependency versions have not been pinned in the repository. For reproducible training and model loading, record and use the package versions from the environment in which the artifacts were generated.

---

## Usage

### Prepare the Input Data

The original input data files are not included in the repository listing.

Before running the training notebook:

1. Open the notebook and inspect its data-loading cells.
2. Identify the original input files and expected file paths.
3. Obtain the required data from its verified source.
4. Place the files in the expected locations or update the paths in the notebook.
5. Confirm the target-label mapping and input feature structure before training.

No external dataset source is specified here because the repository metadata alone does not establish the original dataset's provenance.

### Launch Jupyter Notebook

From the repository root, run:

```bash
jupyter notebook
```

Open the main notebook:

```text
Advanced Exoplanet Classification using Feature Engineering and Ensemble Models.ipynb.ipynb
```

Execute its cells sequentially to run the implemented workflow for data preparation, feature engineering, model experimentation, evaluation, and artifact generation.

### Using the Persisted Model

The repository provides a trained Random Forest classifier and supporting feature artifacts, but it does not contain a standalone inference script or web service.

To use the saved model with new data, the inference workflow must:

1. Load the trusted model and associated artifacts.
2. Validate the structure of the input measurements.
3. Generate the engineered features defined in the notebook.
4. Align the feature columns with the saved feature schema.
5. Apply the required feature selection and scaling operations.
6. Generate class probabilities or predictions using the classifier.
7. Apply the intended classification threshold when converting probabilities into class decisions.

The training notebook should be treated as the reference implementation for the required transformations and label mapping.

---

## Technology Stack

- **Programming Language:** Python
- **Data Processing:** NumPy, Pandas
- **Scientific Computing:** SciPy
- **Machine Learning:** Scikit-learn
- **Tree-Based Models:** Random Forest, XGBoost, CatBoost, LightGBM
- **Ensemble Learning:** Voting and stacking classifiers
- **Signal Analysis:** Fourier Transform and statistical signal features
- **Feature Preprocessing:** StandardScaler and feature-column management
- **Visualization:** Matplotlib, Seaborn
- **Experimentation:** Jupyter Notebook
- **Model Persistence:** Pickle-compatible serialized artifacts
- **Experiment Outputs:** CSV and JSON

---

## Engineering Principles

- Consistent feature definitions between training and inference
- Explicit feature schemas and column ordering
- Correlation-based reduction of redundant features
- Comparative evaluation of individual classifiers and ensembles
- Precision, recall, F1, and ROC-AUC-based performance analysis
- Explicit classification threshold configuration
- Feature-group ablation experiments
- Noise robustness evaluation
- Model artifact persistence and metadata tracking
- Reproducible preprocessing and evaluation workflows

---

## Future Improvements

- Add a version-controlled `requirements.txt` with tested dependency versions.
- Document the verified dataset source and original class-label mapping.
- Package feature engineering, scaling, and classification into a reusable pipeline.
- Add a standalone inference script for new observations.
- Add automated tests for feature extraction, column ordering, and preprocessing consistency.
- Evaluate class-weighting strategies for imbalanced classification.
- Select classification thresholds using a dedicated validation dataset.
- Include precision-recall curves and average precision in model evaluation.
- Reproduce results using repeated stratified splits or cross-validation.
- Validate the pipeline on independent astronomical datasets.
- Add feature-importance analysis and model interpretation using SHAP.
- Track training configurations and model versions.
- Add experiment logging and automated evaluation checks.
- Investigate performance changes under repeated controlled noise experiments.
- Provide an optional FastAPI or Flask interface for programmatic inference.

---

## Limitations

- The original dataset and its source are not documented here because their provenance has not been independently verified.
- Dataset class distribution and label semantics must be confirmed from the notebook before reproducing the experiments.
- High accuracy does not necessarily indicate reliable detection of the positive class.
- Recorded metrics have not been independently reproduced for this README.
- The saved classification threshold differs from the threshold recorded in the model comparison results.
- The repository does not include a standalone inference application.
- Model performance on the recorded dataset does not establish generalization to new astronomical observations.

---

## License

This project is intended to be released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Sushant Kothari**

GitHub: https://github.com/sushantkothari

Repository: https://github.com/sushantkothari/Exoplanet-Classification-Using-Ensemble-Learning
