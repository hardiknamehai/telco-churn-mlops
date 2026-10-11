# Telco Customer Churn Prediction — End-to-End MLOps Pipeline

Predicting telecom customer churn with an XGBoost classifier, served via a FastAPI REST API and an interactive Gradio web UI — fully containerised, tracked with MLflow, and delivered through an automated CI/CD pipeline.

---

## Project Overview & Data Flow

This project solves a binary classification problem: given a telecom customer's demographics, account details, and service subscriptions, predict whether they will churn (leave the company). The model is operationalised end-to-end — from raw data validation through to a live API and web interface.

### Pipeline Stages

```
                       ┌──────────────────────┐
                       │  data/raw/telco.csv   │
                       └──────────┬───────────┘
                                  │
                                  ▼
               ┌──────────────────────────────────┐
               │  1. DATA VALIDATION              │
               │  Great Expectations              │
               │  Schema, ranges, business rules  │
               └──────────────┬───────────────────┘
                              │
                              ▼
               ┌──────────────────────────────────┐
               │  2. PREPROCESSING                │
               │  Drop IDs, coerce types, fill NA │
               └──────────────┬───────────────────┘
                              │
                              ▼
               ┌──────────────────────────────────┐
               │  3. FEATURE ENGINEERING          │
               │  Binary encoding (Yes/No → 0/1)  │
               │  One-hot encoding (drop_first)   │
               │  Bool → int conversion            │
               └──────────────┬───────────────────┘
                              │
                              ▼
               ┌──────────────────────────────────┐
               │  4. MODEL TRAINING & TRACKING    │
               │  XGBoost + scale_pos_weight      │
               │  MLflow: metrics, params, model  │
               │  Artifacts: feature_columns.txt  │
               └──────────────┬───────────────────┘
                              │
                              ▼
               ┌──────────────────────────────────┐
               │  5. SERVING (FastAPI + Gradio)   │
               │  POST /predict  →  JSON response │
               │  GET  /docs     →  Swagger UI    │
               │  GET  /ui       →  Gradio web UI │
               └──────────────────────────────────┘
```

Each stage is a discrete, testable module under `src/`. The serving pipeline applies **identical** feature transformations as training, preventing train-serve skew.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Language** | Python 3.11 |
| **ML Model** | XGBoost 3.0 |
| **Data Processing** | Pandas 2.1, NumPy 1.26, Scikit-learn 1.5 |
| **Experiment Tracking** | MLflow 2.14 (file-based tracking) |
| **Data Validation** | Great Expectations 1.5 |
| **Hyperparameter Tuning** | Optuna 4.4 |
| **REST API** | FastAPI 0.115 + Pydantic 2.8 |
| **Web UI** | Gradio (mounted inside FastAPI) |
| **Server** | Uvicorn 0.30 |
| **Containerisation** | Docker, python:3.11-slim |
| **CI/CD** | GitHub Actions → Docker Hub |

---

## CI/CD & Automated Pipeline Architecture

The project implements both Continuous Integration (CI) and Continuous Training (CT) through two GitHub Actions workflows.

### CI Flow — `ci.yml` (Automated)

```
 git push → main (GitHub)
      │
      ▼
 ┌────────────────────────────────────────────────┐
 │  GitHub Actions  (.github/workflows/ci.yml)    │
 │                                                │
 │  1. Checkout code                              │
 │  2. Set up Python 3.11 + install dependencies  │
 │  3. Lint with Ruff (E, F rules)                │
 │  4. Run automated tests                        │
 │  5. Set up Docker Buildx                       │
 │  6. Login to Docker Hub (secrets)              │
 │  7. docker build + push (multi-arch)           │
 │     → affankhan0966/telco-churn-ml:latest      │
 └────────────────────────────────────────────────┘
```

Every push to `main` triggers linting, testing, and a multi-platform Docker build (linux/amd64 + linux/arm64). The image uses the model at `src/serving/model/production/` — always the current best model. Broken code is blocked before it ever reaches Docker Hub.

### Continuous Training — `retrain.yml` (Manual or Data-Driven)

```
 Click "Run workflow" in GitHub Actions
      │
      ▼
 ┌──────────────────────────────────────────────────┐
 │  GitHub Actions  (.github/workflows/retrain.yml) │
 │                                                  │
 │  1. Set up Python 3.11 + install dependencies    │
 │  2. Run scripts/run_pipeline.py                  │
 │     (validate → preprocess → feature eng → train)│
 │  3. Extract new model ROC AUC from mlruns/       │
 │  4. Compare vs current production model          │
 │     ┌──────────────────────────────────┐         │
 │     │ New ROC > Current ROC ?           │         │
 │     ├── YES → copy to production/       │         │
 │     │         commit + push → ci.yml    │         │
 │     └── NO  → skip, keep current model  │         │
 └──────────────────────────────────────────────────┘
```

**Triggers:**
- Manual: Click **Run workflow** at `Actions → Retrain Model`
- Automatic: Push a new CSV to `data/raw/`

**Evaluation Gate:** The new model is only promoted to production if its ROC AUC exceeds the currently deployed model. This prevents regressions from entering production automatically.

**Required GitHub Secrets:**

| Secret | Purpose |
|---|---|
| `DOCKERHUB_USERNAME` | Docker Hub account username |
| `DOCKERHUB_TOKEN` | Docker Hub personal access token |

---

## Docker Hub — Container Registry

Docker Hub acts as the central **container registry** for this project, decoupling the build process from the deployment process.

```
                      Docker Hub
 GitHub Actions  ──►  ─────────────  ──►  Any Container Runtime
 (build & push)       registry        (pull & run)
                    Image: affankhan0966/
                    telco-churn-ml:latest
```

**Why a container registry matters:**

- **Build once, run anywhere** — the same image works on your laptop, a cloud VM, or a managed service
- **Decouples CI from CD** — GitHub Actions builds and pushes; deployment platforms pull independently
- **Versioned artifacts** — every CI run produces a tagged, immutable image
- **Infrastructure-agnostic** — no vendor lock-in; the image can be pulled by AWS ECS, Render, Fly.io, Hugging Face, or any Docker-compatible platform

The image is always up-to-date: every push to `main` triggers a new build and push, and `retrain.yml` automatically rebuilds when a better model is promoted to production.

---

### Deployment Strategies

The project architecture is purpose-built for deployment portability:

| Design Choice | Deployment Benefit |
|---|---|
| **Containerisation (Docker)** | Single, self-contained artifact — no dependency conflicts |
| **Port exposure (8000)** | Standard HTTP port, works behind any reverse proxy |
| **Health endpoint (`GET /`)** | Any load balancer can verify the app is alive |
| **MLflow model packaging** | Model loaded dynamically at startup, not hardcoded |
| **FastAPI + Gradio in one process** | Single container = simpler orchestration |

#### Current: Local + Docker Pull

The app runs locally for development and demos. Anyone with Docker can pull and run it:

```bash
docker pull affankhan0966/telco-churn-ml:latest
docker run -p 8000:8000 affankhan0966/telco-churn-ml:latest
```

#### Future: AWS ECS Fargate (Enterprise / Production)

For production-grade deployments, the project scales to a fully managed, serverless AWS stack:

```
 GitHub Actions
      │
      ├──►  Amazon ECR  (private container registry)
      │
      └──►  aws ecs update-service  (rolling deployment)
                │
                ▼
 ┌────────────────────────────────────────────────┐
 │  AWS ECS Fargate                               │
 │  ┌──────────────────────────────────────────┐  │
 │  │  Task (container running on port 8000)   │  │
 │  └──────────────────────────────────────────┘  │
 │  • Serverless — no EC2 management              │
 │  • Auto-scaling based on CPU/request count     │
 │  • Runs inside a private VPC                   │
 └───────────────────────┬────────────────────────┘
                         │
                         ▼
 ┌────────────────────────────────────────────────┐
 │  Application Load Balancer (ALB)               │
 │  • Public HTTP:80 → Target Group HTTP:8000     │
 │  • Health checks on GET /                      │
 │  • SSL termination (HTTPS)                     │
 └────────────────────────────────────────────────┘
```

**AWS Components:**

| Service | Role |
|---|---|
| **ECR** | Private, secure Docker image registry (replaces Docker Hub) |
| **ECS Fargate** | Serverless container orchestration — runs the app without managing servers |
| **ALB** | Public entry point, SSL termination, traffic routing, health checks |
| **VPC + Security Groups** | Network isolation; ALB allows inbound 80, tasks allow inbound 8000 from ALB only |
| **CloudWatch** | Centralised logging and monitoring for all services |
| **Auto Scaling** | Scales container count based on traffic (CPU/memory/request thresholds) |

**CD Flow for AWS:**

1. GitHub Actions builds the Docker image
2. Image is pushed to Amazon ECR
3. AWS CLI command updates the ECS service: `aws ecs update-service --force-new-deployment`
4. ECS performs a **rolling update** — starts new tasks, drains old ones, zero downtime

---

## Path to Full Automation & Future Upgrades

This project is currently at **MLOps Level 2+** (automated CI + Continuous Training + containerised serving). Below is the roadmap to reach **Level 3** (full automation with model registry and decoupled architecture).

### 1. Continuous Training (CT) — Implemented

The `retrain.yml` workflow provides one-click model retraining from the GitHub Actions tab. The pipeline trains a new model, evaluates it against the current production model, and auto-promotes only if ROC AUC improves. The promoted model is committed to `src/serving/model/production/`, which triggers `ci.yml` to build a new Docker image.

**Future upgrade:** Add a cron schedule (`schedule: cron: '0 0 * * 0'`) to `retrain.yml` for weekly automated retraining without manual intervention.

### 2. Model Registry & Safety Gates

**Current:** The evaluation gate in `retrain.yml` compares new model ROC AUC against the current production model and auto-promotes only if performance improves — this is a lightweight safety gate without a full model registry.

**Upgrade:** Use the **MLflow Model Registry** with staged transitions:

```
 Experimental → Staging → Production → Archived
```

- **Staging gate:** New model must outperform the current production model on held-out test data
- **Regression tests:** Validate feature schema, prediction distribution, and latency
- **Approval:** Manual approval before promoting to Production (or automated if metrics beat thresholds)

### 3. Decoupled Architecture

**Current:** The model is baked into the Docker image at build time.

**Upgrade:** The Docker image becomes a generic serving container. At startup, it connects to a remote MLflow tracking server and loads the `Production` model dynamically:

```python
model = mlflow.pyfunc.load_model("models:/telco_churn_model/Production")
```

The model artifacts live in a cloud object store (AWS S3 or Cloudflare R2), configured as the MLflow artifact backend. This enables:
- **Instant rollbacks:** Revert to a previous model version via the MLflow UI — no Docker rebuild
- **Independent lifecycles:** Update model without touching application code, and vice versa
- **Zero Git bloat:** No model binaries in the repository

---

## Project Structure

```
Telco-Customer-Churn-ML/
├── .github/workflows/
│   ├── ci.yml                    # CI: Docker build + push to Docker Hub
│   └── retrain.yml               # CT: automated model retraining + evaluation gate
├── scripts/
│   ├── run_pipeline.py           # Main training pipeline (end-to-end)
│   ├── prepare_processed_data.py # Standalone data processing
│   ├── test_pipeline_phase1_data_features.py
│   ├── test_pipeline_phase2_modeling.py
│   └── test_fastapi.py
├── src/
│   ├── data/
│   │   ├── load_data.py          # CSV loader
│   │   └── preprocess.py         # Clean columns, fix types, handle NAs
│   ├── features/
│   │   └── build_features.py     # Binary + one-hot encoding (training)
│   ├── models/
│   │   ├── train.py              # Standalone XGBoost trainer
│   │   ├── tune.py               # Optuna hyperparameter optimisation
│   │   └── evaluate.py           # Classification report + confusion matrix
│   ├── utils/
│   │   ├── validate_data.py      # Great Expectations validation suite
│   │   └── utils.py              # Logger helper
│   ├── app/
│   │   ├── main.py               # FastAPI app + Gradio UI (primary entry point)
│   │   └── app.py                # Alternative entry point
│   └── serving/
│       ├── inference.py          # Production inference pipeline (multi-fallback)
│       └── model/
│           ├── production/       # Current best model (auto-updated by retrain)
│           └── */                # Historical model runs
├── data/
│   ├── raw/
│   │   └── Telco-Customer-Churn.csv  # Dataset (committed for retraining)
│   └── processed/                # Cleaned output
├── dockerfile                    # Docker image (uses production/ model path)
├── requirements.txt              # Python dependencies
└── README.md
```

---

## How to Run Locally

### Prerequisites

- Python 3.11+
- Git
- Docker (optional, for containerised serving)

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/AffanKhan007/Telco-Customer-Churn-ML.git
cd Telco-Customer-Churn-ML

# Create and activate virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Dataset

The Telco Customer Churn dataset is included in the repository at `data/raw/Telco-Customer-Churn.csv`. No download needed.

If you want to update the dataset, replace this file (the retrain workflow is triggered by changes to `data/raw/`).

### 3. Train the Model (Optional)

A pre-trained model is already bundled in `src/serving/model/production/`. To retrain:

```bash
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
```

This runs the full pipeline: data validation → preprocessing → feature engineering → XGBoost training → MLflow logging. The new model and metrics are saved to `mlruns/`.

To promote the new model to production (auto-evaluation gate):

```bash
# Copy the best model to production/
python scripts/run_pipeline.py --input data/raw/Telco-Customer-Churn.csv --target Churn
# Then manually copy the best MLflow run to src/serving/model/production/
# Or use the GitHub Actions retrain.yml workflow which handles this automatically
```

View training runs:

```bash
mlflow ui --backend-store-uri file:./mlruns
```

### 4. Start the API Server

```bash
python -m uvicorn src.app.main:app --host 0.0.0.0 --port 8000
```

### 5. Access the Application

| Endpoint | Description |
|---|---|
| `http://localhost:8000/` | Health check |
| `http://localhost:8000/predict` | REST API — POST JSON, get churn prediction |
| `http://localhost:8000/docs` | Auto-generated Swagger documentation |
| `http://localhost:8000/ui` | Interactive Gradio web form |

### Example API Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "Partner": "No",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "tenure": 1,
    "MonthlyCharges": 85.0,
    "TotalCharges": 85.0
  }'
```

Response:

```json
{"prediction": "Likely to churn"}
```

### 6. Run with Docker

```bash
# Build
docker build -t telco-churn-app -f dockerfile .

# Run
docker run -p 8000:8000 telco-churn-app

# Or pull from Docker Hub
docker pull affankhan0966/telco-churn-ml:latest
docker run -p 8000:8000 affankhan0966/telco-churn-ml:latest
```

### 7. Run Tests

```bash
python scripts/test_pipeline_phase1_data_features.py
python scripts/test_pipeline_phase2_modeling.py
python scripts/test_fastapi.py
```

---

## API Schema

**POST `/predict`** — Predict customer churn

| Field | Type | Description |
|---|---|---|
| `gender` | `string` | `"Male"` or `"Female"` |
| `Partner` | `string` | `"Yes"` or `"No"` |
| `Dependents` | `string` | `"Yes"` or `"No"` |
| `PhoneService` | `string` | `"Yes"` or `"No"` |
| `MultipleLines` | `string` | `"Yes"`, `"No"`, or `"No phone service"` |
| `InternetService` | `string` | `"DSL"`, `"Fiber optic"`, or `"No"` |
| `OnlineSecurity` | `string` | `"Yes"`, `"No"`, or `"No internet service"` |
| `OnlineBackup` | `string` | `"Yes"`, `"No"`, or `"No internet service"` |
| `DeviceProtection` | `string` | `"Yes"`, `"No"`, or `"No internet service"` |
| `TechSupport` | `string` | `"Yes"`, `"No"`, or `"No internet service"` |
| `StreamingTV` | `string` | `"Yes"`, `"No"`, or `"No internet service"` |
| `StreamingMovies` | `string` | `"Yes"`, `"No"`, or `"No internet service"` |
| `Contract` | `string` | `"Month-to-month"`, `"One year"`, or `"Two year"` |
| `PaperlessBilling` | `string` | `"Yes"` or `"No"` |
| `PaymentMethod` | `string` | `"Electronic check"`, `"Mailed check"`, `"Bank transfer (automatic)"`, or `"Credit card (automatic)"` |
| `tenure` | `int` | Number of months as a customer (0-100) |
| `MonthlyCharges` | `float` | Monthly charges in USD |
| `TotalCharges` | `float` | Total charges to date in USD |

**Response:**

```json
{"prediction": "Likely to churn"}
```
or
```json
{"prediction": "Not likely to churn"}
```

---

## Model Performance

The current production XGBoost model (in `src/serving/model/production/`) achieves the following on the held-out test set:

| Metric | Value |
|---|---|
| **ROC AUC** | 0.837 |
| **Precision** | Configurable (threshold 0.35) |
| **Recall** | Configurable (threshold 0.35) |
| **F1 Score** | Configurable (threshold 0.35) |

The classification threshold is set to `0.35` by default (lower than the standard 0.5) to favour recall — catching more potential churners, even at the cost of some false positives. This is adjustable via the `--threshold` flag during training.

**Evaluation Gate:** The `retrain.yml` workflow compares new model ROC AUC against the current production model before promoting. Only models that strictly outperform the current one are deployed.

---

## Author

**Affan Khan** — [GitHub](https://github.com/AffanKhan007)

---

## License

This project is for portfolio and educational purposes. The Telco Customer Churn dataset is publicly available on Kaggle.
