# Demystifying Machine Learning Engineering (MLE) from Scratch

**A hands-on, from-scratch journey into the core of Machine Learning Engineering.**

This repository documents my **ground-up exploration** of Machine Learning Engineering. Instead of relying on prebuilt tools or high-level abstractions, I’m **manually implementing** core ML models, workflows, and architectures—from linear regression and classification to SVMs, neural networks, and beyond.

**Goal:**
To deeply understand *how* models learn, optimize, and generalize. It’s not just about making things work—it’s about making sense of *what’s working* and *why*.


## 📌 Key Features
- **From-scratch implementations** of foundational ML models and algorithms.
- **Focus on intuition and mechanics**: How models learn, optimize, and generalize.
- **No shortcuts**: Raw, transparent, and confidence-driven.

---

## 📂 Directory Structure
   Directory/File       | Description                                                                 |
 |----------------------|-----------------------------------------------------------------------------|
 | `data/`              | Datasets for experiments and model training.                               |
 | `docs/`              | Notes, explanations, and conceptual deep dives.                           |
 | `implementations/`   | Manual implementations of ML models (e.g., regression, classification).   |
 | `kaggle/`            | Kaggle competition exercises and notebooks.                                 |
 | `projects/`          | End-to-end projects (e.g., time series analysis, phishing detection).      |
 | `utils/`             | Reusable utility scripts and helper functions.                             |

---

## 📈 Star History
[![Star History Chart](https://api.star-history.com/svg?repos=Savvythelegend/MLE&type=Date)](https://www.star-history.com/#Savvythelegend/MLE&Date)

---

## Repository Overview

### 1. **Data**
- **Location:** `data/`
  - Example: `heart_failure_clinical_records_dataset.csv`
    A dataset for experimenting with classification and regression models.

### 2. **Implementations**
- **Location:** `implementations/`
  - **Regression:** Manual implementation of logistic regression ([`logistic_regression.ipynb`](implementations/regression/logistic_regression.ipynb)).
  - **Trees:** Decision trees from scratch ([`decision_trees_hands_on.ipynb`](implementations/trees/decision_trees_hands_on.ipynb)).
  - **Dimensionality Reduction:** PCA and SVD implementations ([`principal_component_analysis_from_scratch.ipynb`](implementations/dimensionality_reduction/principal_component_analysis_from_scratch.ipynb)).

### 3. **Kaggle**
- **Location:** `kaggle/intermediateML/`
  - Kaggle exercises: Data leakage, pipelines, XGBoost, and categorical variables.

### 4. **Projects**
- **Location:** `projects/`
  - **Heart Failure Prediction:** End-to-end prediction system ([`Heart_Failure_Prediction.ipynb`](projects/Heart%20Failure%20Prediction%20System/Heart_Failure_Prediction.ipynb)).
  - **Time Series:** Data pipeline for time series analysis ([`hf_data_pipeline.ipynb`](projects/time_series/hf_data_pipeline.ipynb)).

### 5. **Utilities**
- **Location:** `utils/`
  - Reusable helper functions for data loading, preprocessing, and visualization.

---

## How to Use This Repository
1. **Clone the repo:**
   ```bash
   git clone https://github.com/Savvythelegend/MLE.git
