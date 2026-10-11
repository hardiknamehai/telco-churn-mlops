# Telco Customer Churn Prediction — MLOps

An end-to-end machine learning project focused on predicting telecom customer churn and exploring the workflow around model development, experiment tracking, and model serving.

**Status:** Work in progress. The pipeline and setup instructions are being reviewed and tested.

## Overview

Customer churn prediction is a binary classification task: using customer account and service information to estimate whether a customer is likely to leave a telecom provider.

The project includes code for data validation, preprocessing, feature engineering, XGBoost model training, MLflow experiment tracking, and prediction through a FastAPI endpoint with a Gradio interface.

## Workflow

```text
Customer data
    |
    v
Data validation and preprocessing
    |
    v
Feature engineering
    |
    v
XGBoost training and evaluation
    |
    +------> MLflow experiment tracking
    |
    v
Prediction serving
    |
    +------> FastAPI
    |
    +------> Gradio interface

Docker and GitHub Actions are included for containerization
and workflow automation.
```

## Technology Stack

| Area | Tools |
|---|---|
| Language and data processing | Python, Pandas, NumPy |
| Machine learning | XGBoost, scikit-learn |
| Experiment tracking | MLflow |
| Data validation | Great Expectations |
| Hyperparameter tuning | Optuna |
| API | FastAPI, Pydantic, Uvicorn |
| Interface | Gradio |
| Containerization | Docker |
| Automation | GitHub Actions |

## Project Structure

```text
.
├── .github/workflows/    # Automation workflows
├── data/
│   ├── raw/              # Input dataset
│   └── processed/        # Processed data
├── scripts/              # Pipeline and test scripts
├── src/
│   ├── data/             # Data loading and preprocessing
│   ├── features/         # Feature engineering
│   ├── models/           # Training, evaluation, and tuning
│   ├── serving/          # Prediction and model artifacts
│   ├── app/              # FastAPI and Gradio application
│   └── utils/            # Supporting utilities
├── dockerfile
├── requirements.txt
└── README.md
```

## Getting Started

### Requirements

- Python 3.11 or a compatible version
- Git
- Docker (optional)

### 1. Clone the repository

```bash
git clone https://github.com/hardiknamehai/telco-churn-mlops.git
cd telco-churn-mlops
```

### 2. Create an environment and install dependencies

**Windows**

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**macOS/Linux**

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the training pipeline

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```

This script is intended to run data validation, preprocessing, feature engineering, model training, and MLflow logging. Verify the output and dependencies in your environment.

### 4. Start the API

```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

When the application starts successfully, these endpoints are intended to be available:

| Endpoint | Purpose |
|---|---|
| `/` | Health check |
| `/predict` | Customer churn prediction |
| `/docs` | Interactive API documentation |
| `/ui` | Gradio interface |

### 5. Run with Docker

```bash
docker build -t telco-churn-mlops -f dockerfile .
docker run -p 8000:8000 telco-churn-mlops
```

Docker build and runtime behavior should be verified before relying on this method.

## Experiment Tracking and Evaluation

MLflow is included to record experiment parameters, metrics, and model artifacts. The repository also contains model evaluation and hyperparameter-tuning scripts.

Update this section with reproducible evaluation metrics after running the training pipeline and confirming the evaluation procedure. No performance figure is presented here as a verified result yet.

## Current Focus

- Set up the environment and run the project.
- Verify model training and evaluation.
- Test API predictions and the web interface.
- Check Docker build and workflow automation.
- Document verified results and future improvements.

## Reference

The project structure was informed by an existing public implementation: [Telco Customer Churn MLOps reference repository](https://github.com/AffanKhan007/Telco-Customer-Churn-ML).

Before redistributing reused source code, confirm the applicable license or obtain permission. Changes and results specific to this repository should be documented as they are made and tested.
