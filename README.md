<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Scikit--learn-Machine%20Learning-orange?style=for-the-badge&logo=scikitlearn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Random%20Forest-Classification-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Ensemble-Learning-purple?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Signal-Processing-blueviolet?style=for-the-badge"/>
</p>

<h1 align="center">Exoplanet Classification Using Ensemble Learning</h1>

<p align="center">
  Advanced Machine Learning Framework for Exoplanet Detection Using Stellar Light Curves
</p>

<p align="center">
  Statistical Feature Engineering • Fourier Transform Analysis • Correlation-Based Feature Reduction • Ensemble Classification
</p>

---

## Overview

**Exoplanet Classification Using Ensemble Learning** is a machine learning project designed to distinguish stars with confirmed exoplanets from non-exoplanet stars using stellar brightness measurements recorded over time.

The project analyzes stellar light curves derived from the Kepler space telescope dataset. Rather than relying exclusively on raw flux measurements, the pipeline extracts statistical, frequency-domain, peak-based, and signal-variation features to represent the underlying characteristics of each light curve.

The workflow combines feature engineering, correlated-feature removal, feature scaling, and comparative evaluation of multiple classification algorithms. A Random Forest classifier is persisted alongside preprocessing artifacts to support reproducible inference.

The repository demonstrates an end-to-end applied machine learning workflow covering:

- Stellar light-curve data preprocessing
- Statistical and signal-based feature engineering
- Fourier Transform-based frequency analysis
- Peak detection and signal variability analysis
- Correlated-feature identification and removal
- Feature scaling and feature selection
- Comparative evaluation of classical machine learning models
- Weighted voting and stacking ensemble experiments
- Threshold analysis and classification evaluation
- Ablation studies and noise robustness experiments
- Model persistence and metadata export

The project focuses on machine learning-assisted exoplanet classification using preprocessed photometric time-series data.

---

## Why This Project Matters

Detecting exoplanets from stellar observations is challenging because planetary transits can produce subtle changes in the brightness of a host star. These changes may be obscured by observational noise, stellar variability, and other sources of signal variation.

Machine learning can help identify patterns in stellar brightness measurements by learning relationships between extracted signal characteristics and known classification labels.

This project investigates how statistical descriptors, frequency-domain features, signal peaks, and ensemble learning can be used to construct a classification workflow for exoplanet detection.

### Potential Applications

- Exoplanet detection research
- Stellar light-curve classification
- Astronomical time-series analysis
- Machine learning-assisted candidate screening
- Photometric signal characterization
- Feature engineering for scientific datasets
- Comparative evaluation of classification algorithms
- Research and experimentation in computational astrophysics

The model provides predictions based on the patterns present in its training data. It does not independently confirm the existence of an exoplanet or replace astronomical validation.

---

## Key Technical Highlights

- End-to-end supervised machine learning workflow
- Classification using Kepler stellar flux measurements
- Statistical feature extraction from time-series signals
- Fourier Transform-based frequency-domain feature generation
- Peak-based characterization of stellar light curves
- Signal energy and consecutive-difference analysis
- Correlation-based feature reduction
- Standardized feature transformations
- Random Forest classification with persisted model artifacts
- Comparative experiments involving XGBoost, CatBoost, LightGBM, and Logistic Regression
- Weighted voting and stacking ensemble evaluation
- Classification threshold configuration
- Ablation analysis across different feature groups
- Noise robustness evaluation at multiple perturbation levels
- Exported model comparison and experiment results

---

## Research Contributions

- Engineered statistical and signal-processing features from stellar light curves.
- Combined time-domain statistics with frequency-domain and peak-based descriptors.
- Applied correlation analysis to reduce redundant engineered features.
- Compared multiple classical machine learning algorithms using consistent evaluation metrics.
- Evaluated voting and stacking ensembles against individual classifiers.
- Examined the effect of feature groups through ablation experiments.
- Investigated prediction robustness under different levels of injected noise.
- Persisted the trained classifier and supporting preprocessing metadata for subsequent inference.

---

## Core Machine Learning Components

| Component | Purpose |
|---|---|
| Statistical Feature Engineering | Captures the distribution and variability of stellar flux measurements |
| Fourier Transform Analysis | Extracts frequency-domain characteristics from the light curve |
| Peak-Based Features | Represents detected signal peaks and their characteristics |
| Difference-Based Features | Measures changes between consecutive flux observations |
| Correlation Analysis | Identifies redundant engineered features |
| StandardScaler | Standardizes selected feature values |
| Random Forest | Provides the persisted classification model |
| XGBoost | Evaluates gradient-boosted tree classification |
| CatBoost | Evaluates categorical boosting-based classification |
| LightGBM | Evaluates gradient boosting with LightGBM |
| Soft Voting Ensemble | Combines classifier probability estimates |
| Weighted Voting Ensemble | Combines predictions using model weights |
| Stacking Ensemble | Learns a combined prediction from base classifiers |
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
| `Advanced Exoplanet Classification using Feature Engineering and Ensemble Models.ipynb.ipynb` | Main notebook containing data processing, feature engineering, model experimentation, and evaluation |
| `rf_exoplanet_model.pkl` | Serialized Random Forest classification model |
| `feature_scaler.pkl` | Fitted feature scaler used during model preprocessing |
| `raw_feature_columns.pkl` | Persisted raw feature schema used by the pipeline |
| `selected_feature_columns.pkl` | Names of the selected engineered features |
| `dropped_correlated_features.pkl` | Features removed during correlation-based reduction |
| `best_threshold.txt` | Stored classification threshold |
| `pipeline_info.json` | Model metadata, selected features, threshold, and dataset dimensions |
| `model_results.csv` | Comparative model evaluation metrics |
| `ablation_results.csv` | Evaluation results for different feature groups |
| `noise_results.csv` | ROC-AUC measurements from noise robustness experiments |
| `catboost_info/` | CatBoost training logs and metadata |

The repository does not contain a separate inference API or application entry point. The notebook is the primary implementation and experimentation interface.

---

## Dataset

The project uses the Kepler-labelled time-series dataset associated with the Kaggle dataset **Exoplanet Hunting in Deep Space**.

Dataset source:

https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

The dataset contains stellar brightness measurements recorded across successive observation intervals. Each observation represents a star, with its light curve represented by a sequence of flux measurements.

The repository does not list the original training and test CSV files, so they must be obtained separately before reproducing the complete training workflow.

### Dataset Characteristics

| Property | Training Dataset | Test Dataset |
|---|---:|---:|
| Observations | 5,087 | 570 |
| Raw flux measurements per observation | 3,197 | 3,197 |
| Confirmed exoplanet observations | 37 | 5 |
| Non-exoplanet observations | 5,050 | 565 |

The dataset has a significant class imbalance, with confirmed exoplanet observations representing a small fraction of the available examples.

This imbalance is an important consideration when interpreting classification accuracy and selecting an appropriate decision threshold.

### Target Variable

The original dataset uses the following labels:

| Original Label | Meaning |
|---|---|
| `1` | Non-exoplanet star |
| `2` | Star with a confirmed exoplanet |

The notebook's label preprocessing determines how these values are represented internally by the trained classifier.

### Input Data

Each stellar observation contains a sequence of 3,197 flux measurements. These values represent changes in measured stellar brightness over the observation period.

The feature engineering pipeline transforms the raw measurements into a more compact representation of the light curve, enabling conventional machine learning algorithms to operate on extracted statistical and signal-processing features.

---

## System Architecture

### End-to-End Pipeline Overview

1. Load and inspect the stellar light-curve dataset.
2. Separate classification labels from flux measurements.
3. Perform exploratory data analysis and inspect class distribution.
4. Extract statistical descriptors from each flux sequence.
5. Calculate frequency-domain characteristics using Fourier analysis.
6. Generate peak-based and signal-difference features.
7. Identify and remove correlated engineered features.
8. Establish consistent feature ordering and scaling.
9. Train and evaluate multiple classification algorithms.
10. Compare Random Forest, boosting algorithms, and ensemble methods.
11. Evaluate classification metrics and decision thresholds.
12. Perform feature-group ablation experiments.
13. Investigate classification performance under injected noise.
14. Save model artifacts, feature metadata, and evaluation results.

---

## Feature Engineering Methodology

The feature engineering pipeline transforms each stellar light curve into a structured feature vector that captures multiple characteristics of the observed signal.

### Statistical Features

Statistical features summarize the distribution and variability of flux measurements.

| Feature | Description |
|---|---|
| `mean` | Average flux value |
| `std` | Standard deviation of flux measurements |
| `min` | Minimum observed flux |
| `max` | Maximum observed flux |
| `median` | Median flux value |
| `skew` | Measure of distribution asymmetry |
| `kurtosis` | Measure of distribution tail characteristics |
| `q25` | First quartile of flux values |
| `q75` | Third quartile of flux values |
| `energy` | Aggregate signal energy |

These features characterize the overall distribution and variability of a star's observed brightness.

### Frequency-Domain Features

Fourier Transform-based analysis provides information about the frequency composition of the signal.

| Feature | Description |
|---|---|
| `fft_mean` | Mean of the computed frequency-domain representation |
| `fft_peak_freq` | Frequency associated with the dominant detected spectral peak |

Frequency-domain features provide an alternative representation of light-curve patterns that may not be fully characterized by distribution statistics alone.

### Peak-Based Features

Peak-based features characterize prominent structures in the signal.

| Feature | Description |
|---|---|
| `num_peaks` | Number of detected peaks |
| `peak_mean` | Average value of detected peaks |
| `peak_min` | Minimum value among detected peaks |

### Signal-Difference Features

Difference-based features describe changes between consecutive flux measurements.

| Feature | Description |
|---|---|
| `mean_abs_diff` | Mean absolute difference between consecutive observations |
| `diff_energy` | Aggregate energy of the difference signal |
| `zero_crossings` | Number of zero crossings under the implemented feature definition |

Together, these features provide a compact representation of the original stellar brightness sequence.

---

## Feature Selection and Preprocessing

### Correlation-Based Feature Reduction

Highly correlated features can introduce redundancy into a feature set. The project identifies correlated engineered features and persists the dropped feature names in `dropped_correlated_features.pkl`.

The persisted metadata records the following features as removed:

- `range`
- `iqr`
- `mad`
- `rms`
- `fft_std`
- `fft_max`
- `fft_p90`
- `fft_p95`
- `fft_p99`
- `peak_std`
- `peak_max`
- `std_diff`
- `max_abs_diff`

The exact feature-generation and removal procedure is implemented in the notebook.

### Selected Features

The saved pipeline metadata identifies 18 selected engineered features:

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

These features represent statistical properties, frequency characteristics, peak information, and signal variation.

### Feature Scaling

The serialized `feature_scaler.pkl` artifact stores a fitted scaling transformation used by the model workflow.

The same preprocessing sequence, feature definitions, and column ordering must be preserved when applying the model to new observations.

---

## Classification Models

The project evaluates several individual classifiers and ensemble configurations.

| Model | Role |
|---|---|
| Random Forest | Tree-based classification and persisted inference model |
| XGBoost | Gradient-boosted decision tree classification |
| CatBoost | Gradient boosting classification |
| LightGBM | Gradient boosting classification |
| Logistic Regression | Linear classification baseline |
| Soft Voting Ensemble | Combines predicted class probabilities |
| Weighted Voting Ensemble | Combines model predictions using weighted contributions |
| Stacking Ensemble | Combines base classifiers through a meta-estimator |

The saved `pipeline_info.json` identifies Random Forest as the best model according to its recorded ROC-AUC value.

The persisted model artifact is `rf_exoplanet_model.pkl`. The other models appear in the comparative evaluation outputs; the repository listing does not show separate serialized artifacts for each of them.

---

## Evaluation Framework

The project evaluates classification performance using multiple metrics.

| Metric | Purpose |
|---|---|
| Accuracy | Proportion of all observations classified correctly |
| Precision | Proportion of predicted positive observations that are positive |
| Recall | Proportion of actual positive observations detected |
| F1 Score | Harmonic mean of precision and recall |
| ROC-AUC | Measures class discrimination across probability thresholds |

Because confirmed exoplanet observations are rare in the dataset, accuracy alone is insufficient to assess model effectiveness.

Precision, recall, F1 score, and ROC-AUC provide additional information about false-positive behavior, missed positive observations, and ranking performance.

---

## Results Snapshot

The committed `model_results.csv` file contains the following recorded evaluation results. The classification metrics shown below use the threshold values recorded in that file.

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

These results are taken from the committed experiment output and have not been independently reproduced for this README.

### Results Interpretation

- Random Forest has the highest recorded ROC-AUC among the models listed.
- The weighted ensemble records higher accuracy and F1 score than Random Forest at the threshold shown in the results file.
- The ensemble's recall of `0.60` indicates that it detects only a proportion of the positive examples.
- Logistic Regression and LightGBM illustrate why accuracy alone can be misleading when the positive class is rare.
- The small number of positive examples in the test set makes individual metric values sensitive to even a few additional false negatives or true positives.

The evaluation results should therefore be interpreted alongside class imbalance and threshold selection rather than as standalone evidence of generalization.

---

## Classification Threshold

The repository includes a `best_threshold.txt` artifact containing the following threshold:

```text
0.16
```

The threshold determines how predicted probabilities are converted into binary classification decisions.

A lower threshold may increase positive-class recall while also increasing the number of false-positive predictions. Threshold selection should be based on the desired trade-off between precision and recall.

**Reproducibility note:** The saved `model_results.csv` lists a threshold of `0.5` for its model comparisons, whereas `best_threshold.txt` and `pipeline_info.json` record `0.16`. Re-evaluate the saved Random Forest model at the selected threshold before treating the threshold-specific classification metrics as final. The threshold should be selected using validation data rather than optimized directly on the test set.

---

## Ablation Analysis

The repository includes `ablation_results.csv`, which records the performance of different engineered feature groups.

| Feature Set | ROC-AUC | Accuracy | F1 Score |
|---|---:|---:|---:|
| All features | 0.9949 | 0.9947 | 0.5714 |
| Statistical + FFT + Peak + Complexity | 0.9949 | 0.9947 | 0.5714 |
| Statistical only | 0.9912 | 0.9947 | 0.5714 |
| Statistical + FFT + Peak | 0.9885 | 0.9947 | 0.5714 |
| Statistical + FFT | 0.9821 | 0.9947 | 0.5714 |

The recorded experiments compare how different groups of engineered features influence classification performance.

The results suggest that statistical descriptors alone provide substantial discrimination in this experiment, while combining them with additional signal-processing features produces different ROC-AUC values.

These results are specific to the recorded experiment configuration and should not be interpreted as proof that an individual feature group will perform best on other datasets.

---

## Noise Robustness Analysis

The repository contains `noise_results.csv`, which records ROC-AUC values for different noise levels.

| Noise Level | Recorded ROC-AUC |
|---|---:|
| 0.01 | 0.9313 |
| 0.05 | 0.9499 |
| 0.10 | 0.9411 |
| 0.20 | 0.7172 |

The results indicate that the evaluated model's discrimination performance decreases substantially at the highest recorded noise level.

The non-monotonic values at lower noise levels may reflect experimental variability. Further repeated experiments with fixed random seeds and documented perturbation procedures would help establish a more reliable characterization of noise sensitivity.

---

## Model Artifacts

The repository contains the following persisted components.

| Artifact | Description |
|---|---|
| `rf_exoplanet_model.pkl` | Trained Random Forest classifier |
| `feature_scaler.pkl` | Fitted feature scaling transformation |
| `raw_feature_columns.pkl` | Expected raw input feature schema |
| `selected_feature_columns.pkl` | Selected engineered feature names |
| `dropped_correlated_features.pkl` | Features excluded during correlation reduction |
| `best_threshold.txt` | Stored classification threshold |
| `pipeline_info.json` | Model identity, ROC-AUC, threshold, feature metadata, and data dimensions |

For inference, the input must undergo the same feature engineering, column alignment, feature reduction, and scaling sequence used during training.

A serialized estimator alone is not sufficient to reproduce the pipeline if the surrounding transformations are applied differently.

**Security note:** Pickle files can execute arbitrary code during deserialization. Only load model artifacts from trusted sources, and use compatible library versions when loading persisted estimators and scalers.

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

The repository listing does not currently include a `requirements.txt` file. The following command installs the principal libraries used for the documented data processing and model experiments:

```bash
python -m pip install --upgrade pip

pip install numpy pandas scipy matplotlib seaborn scikit-learn joblib xgboost lightgbm catboost jupyter
```

For reproducible experiments, create a `requirements.txt` with the exact dependency versions used to train the saved artifacts.

---

## Usage

### Obtain the Dataset

Download the Kepler-labelled time-series dataset from Kaggle:

https://www.kaggle.com/datasets/keplersmachines/kepler-labelled-time-series-data

Obtain the required training and test CSV files, commonly named:

```text
exoTrain.csv
exoTest.csv
```

Place the files in the locations expected by the notebook. If your directory structure differs from the notebook's configured paths, update the relevant data-loading cells.

### Launch Jupyter Notebook

From the repository root, execute:

```bash
jupyter notebook
```

Open:

```text
Advanced Exoplanet Classification using Feature Engineering and Ensemble Models.ipynb.ipynb
```

Run the cells sequentially to execute the data preparation, feature engineering, model training, evaluation, and artifact-generation workflow.

### Using the Persisted Model

The repository provides a serialized Random Forest model and its supporting feature artifacts. However, it does not currently expose a standalone prediction script or REST API.

To perform reproducible inference:

1. Load the trusted model and preprocessing artifacts.
2. Supply the required raw stellar flux measurements in the expected schema.
3. Apply the same statistical, frequency-domain, peak-based, and difference-based feature engineering.
4. Apply the saved feature-selection and scaling transformations in the training-time order.
5. Generate the classifier's probability or class prediction.
6. Apply the intended classification threshold when converting probabilities into binary decisions.

The exact feature definitions and preprocessing order in the notebook should be treated as the reference implementation.

---

## Technology Stack

- **Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Scientific Computing:** SciPy
- **Machine Learning:** Scikit-learn
- **Tree-Based Models:** Random Forest, XGBoost, CatBoost, LightGBM
- **Ensemble Learning:** Voting and stacking classifiers
- **Signal Processing:** Fourier Transform and statistical time-series analysis
- **Data Visualization:** Matplotlib, Seaborn
- **Model Persistence:** Pickle-compatible artifacts
- **Experimentation:** Jupyter Notebook
- **Result Export:** CSV and JSON

---

## Engineering Principles

- Consistent feature extraction between training and inference
- Explicit feature schemas and column ordering
- Correlation-based reduction of redundant features
- Comparative evaluation of individual models and ensembles
- Precision, recall, and F1 analysis for imbalanced classification
- Decision-threshold configuration
- Ablation experiments for feature-group analysis
- Noise robustness experiments
- Model artifact persistence and metadata tracking
- Reproducible preprocessing and evaluation workflows

---

## Future Improvements

- Add a versioned `requirements.txt` for reproducible environments.
- Document the exact feature engineering and prediction APIs.
- Package preprocessing and classification into a single reusable pipeline.
- Add automated tests for feature ordering and transformation consistency.
- Evaluate class-weighting and resampling strategies for the imbalanced target.
- Perform threshold optimization using a dedicated validation set.
- Report precision-recall curves and average precision alongside ROC-AUC.
- Re-run evaluation with repeated stratified splits or cross-validation.
- Validate results on independent astronomical datasets.
- Quantify feature importance and add model interpretation using SHAP.
- Add a standalone inference script for new stellar light curves.
- Add an optional FastAPI or Flask service for programmatic prediction.
- Track experiments and model versions with a reproducible configuration.
- Expand robustness testing across multiple controlled noise distributions.

---

## Limitations

- Confirmed exoplanet observations form a very small fraction of the training and test datasets.
- High accuracy can occur even when a classifier misses a substantial fraction of positive observations.
- The test set contains only five confirmed exoplanet examples, limiting the stability of its recall and F1 estimates.
- Feature engineering and classification results depend on the input data format and preprocessing implementation.
- The repository does not include the original dataset files or a dedicated production inference service.
- Recorded evaluation metrics have not been independently reproduced for this documentation.
- Successful classification of a historical dataset does not establish performance on new astronomical observations.

---

## Author

**Sushant Kothari**

GitHub: https://github.com/sushantkothari

Repository: https://github.com/sushantkothari/Exoplanet-Classification-Using-Ensemble-Learning
