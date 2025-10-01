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

<custom-element data-json="%7B%22type%22%3A%22table-metadata%22%2C%22attributes%22%3A%7B%22title%22%3A%22Directory%20Structure%22%7D%7D" />

| Directory/File       | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `data/`              | Datasets for experiments and model training.                               |
| `docs/`              | Notes, explanations, and conceptual deep dives.                           |
| `implementations/`   | Manual implementations of ML models (e.g., regression, classification).   |
| `notes/`             | Study notes, code snippets, and external resources.                        |
| `programming/`       | Language-specific resources (Python, SQL).                                  |
| `projects/`          | End-to-end projects (e.g., time series analysis).                           |
| `tests/`             | Unit tests and validation scripts.                                          |

---

## 📈 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Savvythelegend/MLE&type=Date)](https://www.star-history.com/#Savvythelegend/MLE&Date)

---

## Repository Overview

### 1. **Data**
- **Location:** `data/datasets/archive/`
- **Example:** `heart_failure_clinical_records_dataset.csv`
  A dataset for experimenting with classification and regression models.

### 2. **Documentation**
- **Location:** `docs/`
  - `gradients.md`: Deep dive into gradient-based optimization.
  - `linear-nonlinear.md`: Linear vs. nonlinear models—tradeoffs and intuition.
  - `prompt.md`: Notes on prompt engineering for ML workflows.

### 3. **Implementations**
- **Location:** `implementations/`
  - **Regression:** Manual implementation of logistic regression ([`logistic_regression.ipynb`](implementations/regression/logistic_regression.ipynb)).
  - **Upcoming:** SVMs, neural networks, and custom architectures.

### 4. **Notes**
- **Location:** `notes/Machine-Learning-with-Pytorch-Scikit-Learn/`
  - Study notes and utility scripts (e.g., [`plot_decision_regions_script.py`](notes/Machine-Learning-with-Pytorch-Scikit-Learn/utils/plot_decision_regions_script.py)).

### 5. **Programming Resources**
- **Location:** `programming/`
  - **Python:** Intuition-driven notes ([`intuition.md`](programming/python/intuition.md)).
  - **SQL:** Practice queries for data science (e.g., [`Data-Science-Skills_Linkedin.sql`](programming/sql/easy/Data-Science-Skills_Linkedin.sql)).

### 6. **Projects**
- **Location:** `projects/`
  - **Time Series:** End-to-end data pipeline ([`hf_data_pipeline.ipynb`](projects/time_series/hf_data_pipeline.ipynb)).

### 7. **Tests**
- **Location:** `tests/`
  Placeholder for unit tests and validation scripts.

---

## How to Use This Repository
1. **Clone the repo:**
   ```bash
   git clone https://github.com/Savvythelegend/MLE.git
