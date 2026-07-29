# Customer Churn Risk Scoring Prediction System

> Random Forest-based customer churn prediction system with monitoring dashboard, prediction logging, input validation (Pydantic), model integrity check, and automated testing.

![Status](https://img.shields.io/badge/status-production--ready-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

**Random Forest-based customer churn prediction system with monitoring dashboard, prediction logging, input validation (Pydantic), model integrity check, and automated testing.**

**Demo:** Not Applicable · **Documentation:** http://localhost:8000/docs ·

------------------------------------------------------------------------

## 📑 Table of Contents

-   About Project
-   Business Understanding
-   Features
-   Tech Stack
-   Architecture
-   Project Structure
-   Environment Variables
-   API
-   Getting Started
-   Roadmap
-   Testing
-   Metrics
-   AI Components
-   Security
-   Deployment
-   Future Improvements
-   Contributing
-   License
-   Contact

------------------------------------------------------------------------

## 📖 About Project

### Overview

A production-ready Machine Learning system to predict customer churn risk (whether a customer will unsubscribe) using a Random Forest Classifier. It features a high-performance REST API powered by FastAPI, type-safe data validation via Pydantic v2, prediction logging buffered to JSONL files, automatic data drift detection, a real-time visual monitoring dashboard, and automated tests with pytest.

### Objectives

-   Predict customer churn risk in real-time with high accuracy and reliability.
-   Provide API endpoints and a centralized monitoring dashboard to track model performance and data drift status.
-   Guarantee the integrity of the machine learning model during deployment using SHA-256 checksum validation.

### Target Users

-   Data Science / MLOps Engineers looking to test, monitor, and retrain the churn model.
-   Marketing / Customer Relationship Management (CRM) teams to target customer retention strategies effectively.
-   System Administrators / IT Operations to monitor server health and model integrity.

------------------------------------------------------------------------

## 💡 Business Understanding

### Background

The company is experiencing a decline in active subscribers. A predictive classification system is required to identify high-risk churn customers so that the retention team can take proactive and efficient preventive actions.

### Problems & Solutions

| Problem | Solution |
|---|---|
| Unexpected decline in customer retention. | Real-time churn prediction to detect early warning signs of customer dissatisfaction. |
| Deployed model degradation due to shifting customer behavior/data patterns. | Automated data drift detection comparing real-time prediction distribution against training baseline. |
| Risk of model tampering or corruption in the production environment. | SHA-256 checksum validation of the model file during API initialization. |

### Success Metrics (KPI)

-   F1-Score: target >= 0.62 (Current model F1-Score: 0.6269)
-   ROC-AUC: target >= 0.77 (Current model ROC-AUC: 0.7786)
-   Model Integrity Check: 100% verification rate of the model SHA-256 hash

------------------------------------------------------------------------

## ✨ Features

-   Machine Learning Churn Prediction (Random Forest Classifier)
-   Real-time Web-based Monitoring Dashboard Integration (auto-refresh every 20 requests)
-   Secured Metrics API Endpoint (API Key Authentication)
-   Buffered Prediction Logging to JSONL file (minimizes I/O bottleneck)
-   Automated Data Drift Detection (based on prediction target distribution deviation)

------------------------------------------------------------------------

## 🛠 Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| Frontend | HTML, CSS, JS | Visually present monitoring metrics, prediction distributions, data drift alerts, and system health in real-time. |
| Backend | FastAPI, Uvicorn | Provides a high-performance, asynchronous REST API for model inference, health check, and metrics aggregation. |
| Database | JSONL (logs/predictions.jsonl) | Storage for buffered local prediction logs used for metrics computation and audit trails. |
| AI / ML | scikit-learn, pandas, numpy, joblib | Training pipeline orchestration (GridSearchCV), data preprocessing (ColumnTransformer, OneHotEncoder, SimpleImputer), model training (Random Forest), and serialization. |
| Cloud | Not Applicable | Not Applicable |
| DevOps | pytest, httpx, python-dotenv | Automated unit and integration testing of the API, configuration management via .env. |

------------------------------------------------------------------------

## 🏗 Architecture

``` text
┌─────────────────┐      HTTP Request (Data)      ┌─────────────────────────┐
│  Web Dashboard  │ ────────────────────────────> │       FastAPI API       │
│  (Port 3000)    │ <──────────────────────────── │       (Port 8000)       │
└─────────────────┘      JSON Response (Metrics)  └─────────────────────────┘
                                                       │            │
                                         Load Model    │            │ log predictions
                                         & Verify Hash │            ▼
                                                       ▼     ┌──────────────┐
                                                ┌──────────┐ │ predictions. │
                                                │   ML     │ │    jsonl     │
                                                │ Pipeline │ └──────────────┘
                                                └──────────┘
```

------------------------------------------------------------------------

## 📁 Project Structure

``` text
customer_churn_risk_scoring/
├── app/
│   ├── __init__.py
│   └── main.py                 ← FastAPI application entry point
├── dashboard/
│   └── index.html              ← Web monitoring dashboard (HTML/JS)
├── data/
│   └── cth_churn_analysis_train.xlsx  ← Dataset for training
├── logs/
│   └── predictions.jsonl       ← Buffered prediction logs (auto-created)
├── models/
│   ├── final_churn_model.joblib       ← Trained Random Forest model
│   └── final_churn_model_metadata.json ← Model metadata & SHA-256 hash
├── src/
│   ├── __init__.py
│   ├── config.py               ← App settings using Pydantic Settings
│   ├── data_loader.py          ← Data ingestion
│   ├── evaluator.py            ← Model evaluation & integrity validation
│   ├── logger.py               ← Logger utility
│   ├── monitoring.py           ← Metrics & data drift computations
│   ├── pipeline.py             ← End-to-end training pipeline
│   ├── predictor.py            ← Model loader and inference engine
│   ├── preprocessing.py        ← Preprocessing ColumnTransformer
│   ├── schemas.py              ← Pydantic schemas for request/response
│   └── trainer.py              ← Model training & hyperparameter search
├── tests/
│   ├── __init__.py
│   └── test_api.py             ← Automated test suite (pytest)
├── .env                        ← Local environment configuration
├── .gitignore                  ← Git ignored files
├── main.py                     ← Legacy entry point
├── train.py                    ← Standalone training CLI script
├── requirements.txt            ← Python dependencies list
├── README.md                   ← Project documentation (English)
└── README_IDN.md               ← Project documentation (Indonesian)
```

------------------------------------------------------------------------

## ⚙️ Environment Variables

``` env
# API Configuration
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
METRICS_API_KEY=dev-key-change-in-production

# Data Configuration
CHURN_DATA_URL=https://storage.googleapis.com/dqlab-dataset/cth_churn_analysis_train.xlsx
CHURN_DATA_DIR=data
CHURN_LOCAL_DATA_PATH=data/cth_churn_analysis_train.xlsx

# Model Configuration
MODEL_DIR=models
MODEL_PATH=models/final_churn_model.joblib
METADATA_PATH=models/final_churn_model_metadata.json

# Logging Configuration
LOG_DIR=logs
LOG_PATH=logs/predictions.jsonl

# Training Configuration
CHURN_RANDOM_STATE=57
CHURN_TEST_SIZE=0.2
CHURN_CV_SPLITS=5
CHURN_SCORING_METRIC=f1
```

------------------------------------------------------------------------

## 📡 API

| Method | Endpoint | Description |
|---|---|---|
| POST | /predict | Predict customer churn risk |
| GET | /health | Check system health and uptime |
| GET | /metrics | Retrieve MLOps statistics (API Key required) |
| GET | /docs | Interactive Swagger UI docs |

------------------------------------------------------------------------

## 🚀 Getting Started

### Prerequisites

-   Python 3.11 or higher
-   Pip & Virtualenv

### Installation

``` bash
git clone https://github.com/waahyoew/customer-churn-risk-scoring.git
cd customer-churn-risk-scoring

# Set up virtual environment and install dependencies
python -m venv venv
# Windows:
.\venv\Scripts\activate.ps1
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
python train.py

# Run API server (Terminal 1)
uvicorn app.main:app --reload

# Run dashboard server (Terminal 2)
cd dashboard
python -m http.server 3000
```

------------------------------------------------------------------------

## 🗺 Roadmap

-   [ ] Implement React/Next.js dashboard for advanced data visualization.
-   [ ] Automate retraining pipeline using GitHub Actions or Airflow.
-   [ ] Integrate Vector Database for customer feedback sentiment analysis.
-   [ ] Implement data drift alerting via Slack/Email integration.

------------------------------------------------------------------------

## 🧪 Testing

### Unit Test

``` bash
# Run the entire unit test suite
pytest tests/test_api.py -v
```

### Integration Test

``` bash
# Run test coverage for all src and app modules
pytest tests/test_api.py -v --cov=src --cov=app
```

### Performance Test

Not Applicable

### Security Test

``` bash
# Test metrics endpoint security by validating X-API-Key header requirement
pytest tests/test_api.py -k "test_metrics_without_api_key" -v
```

------------------------------------------------------------------------

## 📈 Metrics

| Metric | Target |
|---|---|
| Availability | 99.9% |
| Response Time | < 100ms |
| Throughput | 100 RPS |
| Accuracy | 75.0% |

------------------------------------------------------------------------

## 🤖 AI Components

| Component | Value |
|---|---|
| Model | Random Forest Classifier |
| Embedding | Not Applicable |
| Vector Database | Not Applicable |
| RAG | Not Applicable |
| Agent | Not Applicable |
| Prompting | Not Applicable |
| Evaluation | Stratified K-Fold CV (F1-score & ROC-AUC) |

------------------------------------------------------------------------

## 🔒 Security

-   Authentication: X-API-Key header for metrics endpoint
-   Authorization: Not Applicable
-   Encryption: Model Integrity verification via SHA-256 checksum
-   Secrets Management: Environment variables managed via .env file

------------------------------------------------------------------------

## 🚀 Deployment

| Environment | Target |
|---|---|
| Development | Local Machine (FastAPI + Uvicorn) |
| Staging | Not Applicable |
| Production | GCP / AWS / Docker Container |

------------------------------------------------------------------------

## 🔮 Future Improvements

-   Model hyperparameter optimization using Optuna.
-   Implement load balancing using Nginx to handle high traffic.
-   Persistent database logging (PostgreSQL) instead of local files.

------------------------------------------------------------------------

## 🤝 Contributing

1.  Fork the repository.
2.  Create a feature branch.
3.  Commit using Conventional Commits.
4.  Open a Pull Request.

------------------------------------------------------------------------

## 📄 License

This project is licensed under the MIT License.

------------------------------------------------------------------------

## 📬 Contact

-   Author: Wahyu
-   GitHub: waahyoew
-   Email: wahyuwidihansyah@gmail.com
