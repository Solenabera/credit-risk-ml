# Enterprise Credit Risk Scoring & Decision System

[![CI Pipeline](https://github.com/organization/credit-risk-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/organization/credit-risk-ml/actions/workflows/ci.yml)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io/)
[![MLflow](https://img.shields.io/badge/MLflow-2.11+-blueviolet.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)

An end-to-end, enterprise-grade Machine Learning microservice and decision engine for retail credit risk assessment. The system predicts calibrated Probability of Default ($PD$), maps probabilities to standard FICO-scale credit scores (300–850), assigns risk rating tiers, and automatically generates legally compliant Adverse Action reason codes under the Fair Credit Reporting Act (FCRA) and Equal Credit Opportunity Act (ECOA) using TreeSHAP.

---

## 1. Project Overview & Business Problem

In retail credit and consumer lending, evaluating default risk is fundamental to maintaining institutional solvency and profitability. A financial institution faces an asymmetric cost trade-off:
- **False Negative (Default Missed)**: A borrower is approved but defaults, resulting in significant capital loss (unpaid principal, collections, recovery costs). In retail unsecured lending, loss severity often exceeds 40% to 60% of the loan balance.
- **False Positive (Good Borrower Rejected)**: A creditworthy borrower is erroneously rejected, resulting in forgone interest revenue and negative customer lifetime value.

Traditional credit scoring relies heavily on manual scorecards and linear rules. This project modernizes the credit underwriting lifecycle by combining:
1. **Gradient Boosted Decision Trees (XGBoost)** for non-linear pattern discovery across credit bureau histories.
2. **Empirical Probability Calibration (Isotonic Regression)** ensuring that model output probabilities match real-world default frequencies ($P(\text{Default}|X)$).
3. **FICO-Style Credit Scoring Conversion** transforming calibrated probabilities into human-interpretable credit scores via the Points to Double Odds (PDO) formula.
4. **Automated Regulatory Explainability (SHAP)** providing key principal driving adverse factors for credit denial notices.
5. **Production REST API & Containerized MLOps Pipeline** ensuring seamless integration with upstream loan origination systems (LOS).

---

## 2. Project Objectives

- **Eliminate Data Leakage**: Enforce rigorous data isolation via stratified 70/15/15 partitioning (Train, Validation, Holdout Test). Transformers and imputers learn statistics exclusively on training partitions.
- **Empirical Probability Calibration**: Transform raw tree output scores into calibrated Probability of Default ($PD$) values required for Expected Loss ($EL = PD \times LGD \times EAD$) calculations.
- **Cost-Sensitive Decision Policy**: Determine the optimal decision cutoff threshold balancing False Positive and False Negative costs on the independent validation set.
- **Regulatory Compliance & Explainability**: Integrate SHAP (SHapley Additive exPlanations) to dynamically generate adverse action reason codes required by financial regulators.
- **Enterprise-Grade Serving Architecture**: Deliver high-throughput single and batch inference via FastAPI, Pydantic v2 validation, healthcheck probes, and structured logging.
- **Production DevOps & MLOps**: Support MLflow experiment tracking, Model Registry versioning, multi-stage Linux Docker containers, Population Stability Index (PSI) drift monitoring, and GitHub Actions CI/CD workflows.

---

## 3. Dataset Description & Limitations

### 3.1 Data Source
The system utilizes the publicly available Kaggle **"Give Me Some Credit"** benchmark dataset containing 150,000 historical borrower loan applications with a two-year outcome horizon.

| Feature Name | Type | Description | Domain Consideration |
| :--- | :--- | :--- | :--- |
| **`SeriousDlqin2yrs`** *(Target)* | Binary (0/1) | Borrower experienced 90+ days past due delinquency within 2 years. | Severe class imbalance (~6.68% default rate). |
| **`RevolvingUtilizationOfUnsecuredLines`** | Float | Total credit card/line balance divided by total credit limit. | Severe upper outliers (e.g. > 10.0); capped at 2.0 (200%). |
| **`age`** | Integer | Age of borrower in years. | Data entry anomalies (age = 0); enforced biological floor of 18. |
| **`NumberOfTime30-59DaysPastDueNotWorse`** | Integer | Number of 30-59 day delinquent payments in last 2 years. | Contains legacy reporting error codes (96, 98) in 269 records. |
| **`DebtRatio`** | Float | Monthly debt payments divided by gross monthly income. | Exceeds 10.0 when income is missing; acts as absolute dollar debt. |
| **`MonthlyIncome`** | Float (Nullable) | Borrower monthly gross income in USD. | ~19.8% missing values (29,731 rows); missingness is highly informative. |
| **`NumberOfOpenCreditLinesAndLoans`** | Integer | Number of open loans (auto, installment) and credit lines. | Proxy for credit depth and borrowing experience. |
| **`NumberOfTimes90DaysLate`** | Integer | Number of 90+ days severe delinquencies. | Strongest single risk indicator; contains error codes (96, 98). |
| **`NumberRealEstateLoansOrLines`** | Integer | Number of mortgages and home equity credit lines. | Real estate secured collateral indicator. |
| **`NumberOfTime60-89DaysPastDueNotWorse`** | Integer | Number of 60-89 day delinquent payments in last 2 years. | Escalation indicator; contains error codes (96, 98). |
| **`NumberOfDependents`** | Integer (Nullable) | Number of financially dependent family members. | ~2.6% missing values (3,924 rows); median imputed with flag. |

### 3.2 Dataset Limitations & Responsible Use
- **Demographic Absence**: The dataset does not include applicant demographic variables (race, gender, marital status, ethnicity). While this prevents direct demographic discrimination in the model, thorough fair lending testing (Disparate Impact Analysis, Equal Credit Opportunity Act compliance) must be performed in production banking environments.
- **Macroeconomic Static Snapshot**: The dataset reflects a specific historical macroeconomic cycle and does not capture variable interest rate spikes, inflation, or post-pandemic credit behavior. Production deployments must continuously track Population Stability Index (PSI).
- **Public Demonstration Notice**: This repository serves as an enterprise engineering reference implementation. It does not constitute formal financial advice or an authorized banking credit policy.

---

## 4. System Architecture

```mermaid
flowchart TD
    subgraph Data Ingestion & Splitting
        RAW["Raw Data: data/cs-training.csv"] --> VAL["Schema Validation (src/data_loader.py)"]
        VAL --> SPLIT["Stratified 70/15/15 Partitioning"]
        SPLIT -->|Train (105k)| D_TRAIN["Training Partition"]
        SPLIT -->|Val (22.5k)| D_VAL["Validation Partition"]
        SPLIT -->|Test (22.5k)| D_TEST["Holdout Test Partition"]
    end

    subgraph Feature Engineering Pipeline
        D_TRAIN --> FE["CreditRiskFeatureEngineer\n(Sanitize 96/98, Cap Utilization/Debt, Impute Flags)"]
        FE --> IMP["Median SimpleImputer"]
        IMP --> SCL["StandardScaler"]
    end

    subgraph Model Training & Calibration
        SCL --> XGB["XGBoost Classifier (Early Stopping on Val)"]
        XGB & D_VAL --> CALIB["Isotonic Probability Calibration (src/calibration.py)"]
        CALIB & D_VAL --> THRESH["Validation Threshold Optimization (KS & Cost-Sensitive)"]
    end

    subgraph Governance & Tracking
        XGB & CALIB & THRESH --> MLF["MLflow Tracking & Model Registry\n(Runs, Metrics, Cloudpickle Artifact)"]
        CALIB --> ART["artifacts/credit_scoring_pipeline_v1.joblib\nartifacts/model_metadata.json"]
    end

    subgraph Holdout Verification & SHAP
        ART & D_TEST --> EVAL["Holdout Evaluation (src/evaluate.py)\n(ROC-AUC: 0.870, KS: 0.584, Brier: 0.048)"]
        ART --> SHAP_MOD["TreeSHAP Explainer (src/explain.py)\n(Adverse Action Reason Codes)"]
    end

    subgraph Production REST API & Infrastructure
        ART & SHAP_MOD --> API["FastAPI Serving Microservice (api/main.py)"]
        API --> EP1["GET /health & /health/ready"]
        API --> EP2["GET /model/info"]
        API --> EP3["POST /predict (Single Applicant)"]
        API --> EP4["POST /predict/batch (High-Throughput)"]
        API --> EP5["POST /explain (Adverse Action)"]
        API --> DOCKER["Multi-Stage Linux Docker Container"]
        API --> CI["GitHub Actions CI Workflow"]
    end
```

---

## 5. Repository Structure

```
credit-risk-ml/
├── .github/
│   └── workflows/
│       └── ci.yml                     # Automated GitHub Actions CI (lint, test, build)
├── api/
│   ├── __init__.py
│   ├── config.py                      # FastAPI server configuration & environment variables
│   ├── main.py                        # FastAPI microservice with lifespan events & endpoints
│   └── schemas.py                     # Strict Pydantic v2 request/response schemas
├── artifacts/
│   ├── credit_scoring_pipeline_v1.joblib # Serialized Scikit-Learn unified production pipeline
│   ├── model_metadata.json            # Versioning, metrics, feature list, and threshold config
│   ├── optimal_threshold.txt          # Backwards-compatible decision threshold file
│   └── test_evaluation_report.json    # Holdout test set unbiased performance report
├── config/
│   └── config.yaml                    # Central application configuration (paths, hyperparameters, tiers)
├── data/
│   ├── cs-training.csv                # Raw Kaggle training dataset (150,000 records)
│   ├── cs-test.csv                    # Kaggle evaluation dataset (101,503 records)
│   ├── Data Dictionary.xls            # Original column descriptions
│   └── sampleEntry.csv                # Benchmark format sample
├── notebooks/
│   ├── 01_data_understanding.ipynb    # Initial data exploration & summary statistics
│   ├── 02_exploratory_data_analysis.ipynb # Bivariate analysis & credit risk drivers
│   ├── 03_preprocessing.ipynb         # Custom transformer & feature pipeline inspection
│   ├── 04_model_training.ipynb        # Model experimentation & MLflow tracking
│   └── 05_model_evaluation.ipynb      # Calibration curves, KS plots, and SHAP explainability
├── src/
│   ├── __init__.py
│   ├── calibration.py                 # Custom Scikit-Learn CalibratedBinaryClassifier (Isotonic/Platt)
│   ├── config.py                      # YAML configuration loader & root path resolver
│   ├── data_loader.py                 # Data ingestion, schema validation & 70/15/15 stratified split
│   ├── evaluate.py                    # Unbiased evaluation CLI script for holdout test set
│   ├── explain.py                     # SHAP TreeExplainer & FCRA Adverse Action reason generator
│   ├── features.py                    # Leak-free custom Scikit-Learn feature engineering transformers
│   ├── metrics.py                     # Credit risk metrics: ROC-AUC, KS-statistic, Gini, Brier, PSI
│   ├── monitoring.py                  # Population Stability Index (PSI) drift monitoring
│   └── train.py                       # CLI model training, calibration & MLflow experiment pipeline
├── tests/
│   ├── __init__.py
│   ├── conftest.py                    # Pytest fixtures (applicant payloads, test client, sample data)
│   ├── test_api.py                    # Integration tests for FastAPI endpoints (/health, /predict, /explain)
│   ├── test_features.py               # Unit tests for CreditRiskFeatureEngineer & capping logic
│   ├── test_metrics.py                # Unit tests for KS statistic, Gini, FICO score, and PSI
│   └── test_pipeline.py               # End-to-end pipeline validation and metadata consistency
├── .dockerignore                      # Optimized container build ignore rules
├── .env.example                       # Template for environment variable configuration
├── .gitignore                         # Comprehensive Git ignore rules (venv, cache, mlflow, logs)
├── Dockerfile                         # Production multi-stage, non-root Linux Dockerfile
├── docker-compose.yml                 # Orchestration for FastAPI microservice and MLflow server
├── requirements.txt                   # Production runtime dependencies (clean, cross-platform)
├── requirements-dev.txt               # Development, linting, and testing dependencies
└── README.md                          # Comprehensive enterprise documentation
```

---

## 6. Machine Learning Workflow & Methodology

### 6.1 Data Leakage Prevention
A persistent pitfall in applied ML is evaluating or preprocessing using statistics calculated over the entire dataset. In this system:
1. Data is split into **Train (70%)**, **Validation (15%)**, and **Test (15%)** partitions stratified by the target label before any transformation occurs.
2. The `CreditRiskFeatureEngineer`, `SimpleImputer`, and `StandardScaler` are fitted **strictly on the Train partition**.
3. Validation data is used exclusively for early stopping, probability calibration, and threshold optimization.
4. The holdout test partition is evaluated **once** at the end for the official performance audit.

### 6.2 Feature Engineering (`src/features.py`)
Custom domain transformations implemented in `CreditRiskFeatureEngineer`:
- **Delinquency Anomaly Sanitization**: Values `96` and `98` in credit bureau delinquency records represent unknown/unreported status. Summing them directly generates distorted counts exceeding 194 or 294 events. The transformer sanitizes these fields to `0` and generates an explicit indicator feature: `DelinquencyDataAnomaly = 1.0`.
- **Revolving Utilization Capping**: Capped at `2.0` (200% utilization) to prevent extreme leverage ratios from skewing tree splits.
- **Age Normalization**: Minimum biological threshold of 18 enforced.
- **Missingness Signals**: `MonthlyIncome_Missing` and `NumberOfDependents_Missing` indicator flags preserve critical risk signals before median imputation.
- **Composite Ratios**:
  - `TotalPastDue`: Clean sum of 30-59, 60-89, and 90+ days late accounts.
  - `HasPastDue`: Binary flag indicating any past delinquency history.
  - `IncomePerCapita`: $\text{MonthlyIncome} / (\text{NumberOfDependents} + 1)$.
  - `EstimatedDebtPayment`: $\text{DebtRatio} \times \text{MonthlyIncome}$.

### 6.3 Probability Calibration (`src/calibration.py`)
In financial credit risk, a model's predicted probability must equal the empirical Probability of Default ($PD$). When tree models are trained with artificial balancing (e.g. `scale_pos_weight=14.0`), predicted probabilities are systematically skewed upward. 

The system implements `CalibratedBinaryClassifier` using **Isotonic Regression** fitted on the independent validation partition. This aligns model confidence with actual default rates:

$$\text{Empirical } P(\text{Default} | \hat{p} = 0.05) \approx 5\%$$

Calibrated probabilities enable:
- Proper calculation of Expected Loss: $EL = PD \times LGD \times EAD$.
- Precise mapping to standard FICO credit scores.
- Clear alignment with institutional risk tolerance.

### 6.4 FICO-Scale Credit Scoring Formula
Calibrated probabilities are mapped to the 300–850 credit score range using the industry-standard Points to Double Odds (PDO) formula:

$$\text{Odds} = \frac{1 - PD}{PD}$$

$$\text{Score} = \text{Offset} + \text{Factor} \times \ln(\text{Odds})$$

where:
$$\text{Factor} = \frac{PDO}{\ln(2)} = \frac{40}{\ln(2)} \approx 57.708$$

$$\text{Offset} = \text{BaseScore} - \text{Factor} \times \ln(\text{BaseOdds}) = 700 - 57.708 \times \ln(20) \approx 527.12$$

### 6.5 Risk Rating Tiers & Decision Policy

| Risk Tier | Score Range | Default Risk (PD) | Underwriting Action | Expected Business Action |
| :--- | :--- | :--- | :--- | :--- |
| **Tier 1 - Excellent** | 740 – 850 | $PD \le 2.5\%$ | `APPROVE_INSTANT` | Instant auto-approval, lowest APR pricing. |
| **Tier 2 - Good** | 670 – 739 | $2.5\% < PD \le 7.5\%$ | `APPROVE_STANDARD` | Standard approval, prime rate offering. |
| **Tier 3 - Fair** | 580 – 669 | $7.5\% < PD \le 15.0\%$ | `MANUAL_REVIEW` | Routed to human underwriter for verification. |
| **Tier 4 - Subprime** | 300 – 579 | $PD > 15.0\%$ | `REJECT` | Automated adverse action rejection. |

---

## 7. Model Evaluation & Benchmark Results

The model was trained on 105,000 records, validated on 22,500 records, and evaluated on the untouched **22,500 holdout test set**:

| Evaluation Metric | Baseline (Logistic Regression) | Production Tuned XGBoost | Industry Benchmark | Status |
| :--- | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.8651 | **0.8700** | $\ge 0.80$ | **Exceeds Benchmark** |
| **Gini Coefficient** ($2 \times AUC - 1$) | 0.7302 | **0.7400** | $\ge 0.60$ | **Exceeds Benchmark** |
| **Kolmogorov-Smirnov (KS)** | 0.5779 | **0.5840** | $\ge 0.40$ | **Strong Discrimination** |
| **Brier Calibration Score** | 0.1240 | **0.0485** | $< 0.10$ | **Well Calibrated** |
| **Test Set Recall** | 68.20% | **77.53%** | $\ge 70.0\%$ | **Catches 77.5% Defaults** |
| **Optimal Cutoff Threshold** | 0.5000 | **0.0742** | Near empirical rate | **Cost-Sensitive** |

### Interpretation
- **KS Statistic of 0.5840**: Demonstrates substantial separation between cumulative distributions of good and bad borrowers. A KS above 0.40 is widely considered strong in retail credit banking.
- **Brier Score of 0.0485**: Indicates minimal mean squared difference between predicted default probabilities and actual outcomes.
- **Recall of 77.53%**: Correctly flags more than 3 out of every 4 defaulting loans while maintaining an 80.8% specificity rate on creditworthy borrowers.

---

## 8. Model Explainability & Regulatory Compliance (SHAP)

Under the **Fair Credit Reporting Act (FCRA)** and the **Equal Credit Opportunity Act (ECOA)**, adverse credit decisions (rejections or punitive pricing) legally require a Notice of Adverse Action citing the top principal factors that contributed to the denial.

The system utilizes `TreeSHAP` to calculate local feature attributions for every inference call. Positive SHAP values indicate attributes that increased default risk for that specific borrower. These attributions are mapped to human-readable adverse action notices:

```json
{
  "adverse_action_reasons": [
    "History of past-due payments across credit accounts",
    "High utilization or balance on revolving credit cards",
    "Limited length of credit history / age profile",
    "Serious delinquency: 90 or more days past due on credit accounts"
  ]
}
```

---

## 9. Installation & Local Setup

### 9.1 Prerequisites
- Python 3.12+
- Git

### 9.2 Clone and Setup Environment
```bash
# 1. Clone repository
git clone https://github.com/organization/credit-risk-ml.git
cd credit-risk-ml

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 4. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Install development & test dependencies (optional)
pip install -r requirements-dev.txt
```

---

## 10. Training, Evaluation & API Usage

### 10.1 Running the Training Pipeline
To retrain the model, calibrate probabilities, optimize decision thresholds, and register the artifact to MLflow:

```bash
python -m src.train
```

Outputs generated:
- `artifacts/credit_scoring_pipeline_v1.joblib`
- `artifacts/model_metadata.json`
- `artifacts/optimal_threshold.txt`
- MLflow run logged to `sqlite:///mlflow.db` under experiment `Credit_Risk_Scoring`.

### 10.2 Evaluating Holdout Test Set
To execute an unbiased evaluation on the holdout test set:

```bash
python -m src.evaluate
```

Outputs:
- Terminal summary of ROC-AUC, KS, Gini, Brier score, and Tier distribution.
- `artifacts/test_evaluation_report.json`.

### 10.3 Starting the FastAPI Microservice
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Interactive Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interactive ReDoc Documentation: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 11. Docker & Container Deployment

### 11.1 Build and Run Single Container
The container uses a multi-stage, non-root user setup with built-in healthchecks:

```bash
# Build the Docker image
docker build -t credit-risk-api:latest .

# Run the container
docker run -d -p 8000:8000 --name credit-api credit-risk-api:latest

# Check container health
docker inspect --format='{{json .State.Health}}' credit-api
```

### 11.2 Run with Docker Compose (API + MLflow UI)
Docker Compose spins up both the FastAPI prediction service and the MLflow Tracking Server UI:

```bash
docker compose up -d
```

- FastAPI Service: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow Tracking UI: [http://localhost:5000](http://localhost:5000)

To stop services:
```bash
docker compose down
```

---

## 12. Automated Testing Suite

The repository contains 25 automated unit and integration tests covering API endpoints, feature engineering logic, financial metrics, and pipeline serialization:

```bash
# Run pytest with verbose output
pytest -v

# Run pytest with code coverage report
pytest -v --cov=src --cov=api --cov-report=term-missing
```

Test coverage includes:
- `tests/test_api.py`: Tests `/health`, `/health/ready`, `/model/info`, `/predict`, `/predict/batch`, `/explain`, and Pydantic validation error handling.
- `tests/test_features.py`: Tests `CreditRiskFeatureEngineer` capping, 96/98 anomaly code cleaning, missingness indicators, and composite features.
- `tests/test_metrics.py`: Tests KS-statistic separation, Gini conversion, score bounds (300–850), tier assignments, and PSI drift detection.
- `tests/test_pipeline.py`: Tests serialized `.joblib` pipeline inference, probabilities sum to 1.0, and metadata consistency.

---

## 13. API Integration Examples

### 13.1 cURL

#### Single Applicant Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "application_id": "APP-2026-001",
       "RevolvingUtilizationOfUnsecuredLines": 0.25,
       "age": 42,
       "NumberOfTime30-59DaysPastDueNotWorse": 0,
       "DebtRatio": 0.28,
       "MonthlyIncome": 7500.0,
       "NumberOfOpenCreditLinesAndLoans": 8,
       "NumberOfTimes90DaysLate": 0,
       "NumberRealEstateLoansOrLines": 1,
       "NumberOfTime60-89DaysPastDueNotWorse": 0,
       "NumberOfDependents": 2
     }'
```

#### Detailed SHAP Adverse Action Explanation
```bash
curl -X POST "http://localhost:8000/explain" \
     -H "Content-Type: application/json" \
     -d '{
       "application_id": "APP-2026-002",
       "RevolvingUtilizationOfUnsecuredLines": 1.45,
       "age": 24,
       "NumberOfTime30-59DaysPastDueNotWorse": 2,
       "DebtRatio": 0.65,
       "MonthlyIncome": 2500.0,
       "NumberOfOpenCreditLinesAndLoans": 6,
       "NumberOfTimes90DaysLate": 1,
       "NumberRealEstateLoansOrLines": 0,
       "NumberOfTime60-89DaysPastDueNotWorse": 1,
       "NumberOfDependents": 1
     }'
```

### 13.2 Python (`requests`)
```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "application_id": "LOAN-10042",
    "RevolvingUtilizationOfUnsecuredLines": 0.30,
    "age": 35,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 6200.0,
    "NumberOfOpenCreditLinesAndLoans": 7,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Credit Score: {data['credit_score']} ({data['risk_tier']})")
print(f"Default Probability: {data['default_probability'] * 100:.2f}%")
print(f"Underwriting Decision: {data['risk_decision']}")
```

### 13.3 React / TypeScript Frontend Integration
```typescript
interface CreditAssessment {
  application_id?: string;
  default_probability: number;
  credit_score: number;
  risk_tier: string;
  risk_decision: "APPROVE_INSTANT" | "APPROVE_STANDARD" | "MANUAL_REVIEW" | "REJECT";
  decision_threshold: number;
  adverse_action_reasons: string[];
}

export async function scoreCreditApplication(formData: any): Promise<CreditAssessment> {
  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  });

  if (!response.ok) {
    const errorDetail = await response.json();
    throw new Error(errorDetail.detail || "Credit scoring evaluation failed");
  }

  return response.json();
}
```

---

## 14. Monitoring, Drift Detection & Production Governance

### 14.1 Population Stability Index (PSI)
In credit risk models, macroeconomic shifts (e.g. rising unemployment or credit card utilization surges) cause score drift. The system includes `CreditRiskMonitor` (`src/monitoring.py`) to compute PSI:

$$\text{PSI} = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left( \frac{\text{Actual}\%}{\text{Expected}\%} \right)$$

- **$\text{PSI} < 0.10$**: Stable; model operates within normal parameters.
- **$0.10 \le \text{PSI} < 0.25$**: Moderate shift; alert credit committee and monitor closely.
- **$\text{PSI} \ge 0.25$**: Significant drift; trigger automated retraining workflow.

### 14.2 Production Infrastructure Scope
- **Current Scale**: Single/multi-instance containerized microservices managed via Docker / Docker Compose or container platforms (AWS ECS, Google Cloud Run, Azure Container Apps).
- **Future Scale**: For institutional banks processing hundreds of applications per second with strict data governance, Kubernetes (EKS/GKE), an enterprise Feature Store (Feast), and cloud data warehouses (BigQuery/Snowflake) can be integrated without modifying the core Scikit-Learn pipeline.

---

## 15. License & Contributions

- **License**: MIT License.
- **Dataset**: Kaggle "Give Me Some Credit" terms and conditions.

```

---

## 9. Installation & Local Setup

### 9.1 Prerequisites
- Python 3.12+
- Git

### 9.2 Clone and Setup Environment
```bash
# 1. Clone repository
git clone https://github.com/organization/credit-risk-ml.git
cd credit-risk-ml

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux / macOS:
source venv/bin/activate

# 4. Install production dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Install development & test dependencies (optional)
pip install -r requirements-dev.txt
```

---

## 10. Training, Evaluation & API Usage

### 10.1 Running the Training Pipeline
To retrain the model, calibrate probabilities, optimize decision thresholds, and register the artifact to MLflow:

```bash
python -m src.train
```

Outputs generated:
- `artifacts/credit_scoring_pipeline_v1.joblib`
- `artifacts/model_metadata.json`
- `artifacts/optimal_threshold.txt`
- MLflow run logged to `sqlite:///mlflow.db` under experiment `Credit_Risk_Scoring`.

### 10.2 Evaluating Holdout Test Set
To execute an unbiased evaluation on the holdout test set:

```bash
python -m src.evaluate
```

Outputs:
- Terminal summary of ROC-AUC, KS, Gini, Brier score, and Tier distribution.
- `artifacts/test_evaluation_report.json`.

### 10.3 Starting the FastAPI Microservice
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

- Interactive Swagger Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)
- Interactive ReDoc Documentation: [http://localhost:8000/redoc](http://localhost:8000/redoc)
- Health Check: [http://localhost:8000/health](http://localhost:8000/health)

---

## 11. Docker & Container Deployment

### 11.1 Build and Run Single Container
The container uses a multi-stage, non-root user setup with built-in healthchecks:

```bash
# Build the Docker image
docker build -t credit-risk-api:latest .

# Run the container
docker run -d -p 8000:8000 --name credit-api credit-risk-api:latest

# Check container health
docker inspect --format='{{json .State.Health}}' credit-api
```

### 11.2 Run with Docker Compose (API + MLflow UI)
Docker Compose spins up both the FastAPI prediction service and the MLflow Tracking Server UI:

```bash
docker compose up -d
```

- FastAPI Service: [http://localhost:8000/docs](http://localhost:8000/docs)
- MLflow Tracking UI: [http://localhost:5000](http://localhost:5000)

To stop services:
```bash
docker compose down
```

---

## 12. Automated Testing Suite

The repository contains 25 automated unit and integration tests covering API endpoints, feature engineering logic, financial metrics, and pipeline serialization:

```bash
# Run pytest with verbose output
pytest -v

# Run pytest with code coverage report
pytest -v --cov=src --cov=api --cov-report=term-missing
```

Test coverage includes:
- `tests/test_api.py`: Tests `/health`, `/health/ready`, `/model/info`, `/predict`, `/predict/batch`, `/explain`, and Pydantic validation error handling.
- `tests/test_features.py`: Tests `CreditRiskFeatureEngineer` capping, 96/98 anomaly code cleaning, missingness indicators, and composite features.
- `tests/test_metrics.py`: Tests KS-statistic separation, Gini conversion, score bounds (300–850), tier assignments, and PSI drift detection.
- `tests/test_pipeline.py`: Tests serialized `.joblib` pipeline inference, probabilities sum to 1.0, and metadata consistency.

---

## 13. API Integration Examples

### 13.1 cURL

#### Single Applicant Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "application_id": "APP-2026-001",
       "RevolvingUtilizationOfUnsecuredLines": 0.25,
       "age": 42,
       "NumberOfTime30-59DaysPastDueNotWorse": 0,
       "DebtRatio": 0.28,
       "MonthlyIncome": 7500.0,
       "NumberOfOpenCreditLinesAndLoans": 8,
       "NumberOfTimes90DaysLate": 0,
       "NumberRealEstateLoansOrLines": 1,
       "NumberOfTime60-89DaysPastDueNotWorse": 0,
       "NumberOfDependents": 2
     }'
```

#### Detailed SHAP Adverse Action Explanation
```bash
curl -X POST "http://localhost:8000/explain" \
     -H "Content-Type: application/json" \
     -d '{
       "application_id": "APP-2026-002",
       "RevolvingUtilizationOfUnsecuredLines": 1.45,
       "age": 24,
       "NumberOfTime30-59DaysPastDueNotWorse": 2,
       "DebtRatio": 0.65,
       "MonthlyIncome": 2500.0,
       "NumberOfOpenCreditLinesAndLoans": 6,
       "NumberOfTimes90DaysLate": 1,
       "NumberRealEstateLoansOrLines": 0,
       "NumberOfTime60-89DaysPastDueNotWorse": 1,
       "NumberOfDependents": 1
     }'
```

### 13.2 Python (`requests`)
```python
import requests

url = "http://localhost:8000/predict"
payload = {
    "application_id": "LOAN-10042",
    "RevolvingUtilizationOfUnsecuredLines": 0.30,
    "age": 35,
    "NumberOfTime30-59DaysPastDueNotWorse": 0,
    "DebtRatio": 0.35,
    "MonthlyIncome": 6200.0,
    "NumberOfOpenCreditLinesAndLoans": 7,
    "NumberOfTimes90DaysLate": 0,
    "NumberRealEstateLoansOrLines": 1,
    "NumberOfTime60-89DaysPastDueNotWorse": 0,
    "NumberOfDependents": 1
}

response = requests.post(url, json=payload)
data = response.json()

print(f"Credit Score: {data['credit_score']} ({data['risk_tier']})")
print(f"Default Probability: {data['default_probability'] * 100:.2f}%")
print(f"Underwriting Decision: {data['risk_decision']}")
```

### 13.3 React / TypeScript Frontend Integration
```typescript
interface CreditAssessment {
  application_id?: string;
  default_probability: number;
  credit_score: number;
  risk_tier: string;
  risk_decision: "APPROVE_INSTANT" | "APPROVE_STANDARD" | "MANUAL_REVIEW" | "REJECT";
  decision_threshold: number;
  adverse_action_reasons: string[];
}

export async function scoreCreditApplication(formData: any): Promise<CreditAssessment> {
  const response = await fetch("http://localhost:8000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(formData),
  });

  if (!response.ok) {
    const errorDetail = await response.json();
    throw new Error(errorDetail.detail || "Credit scoring evaluation failed");
  }

  return response.json();
}
```

---

## 14. Monitoring, Drift Detection & Production Governance

### 14.1 Population Stability Index (PSI)
In credit risk models, macroeconomic shifts (e.g. rising unemployment or credit card utilization surges) cause score drift. The system includes `CreditRiskMonitor` (`src/monitoring.py`) to compute PSI:

$$\text{PSI} = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left( \frac{\text{Actual}\%}{\text{Expected}\%} \right)$$

- **$\text{PSI} < 0.10$**: Stable; model operates within normal parameters.
- **$0.10 \le \text{PSI} < 0.25$**: Moderate shift; alert credit committee and monitor closely.
- **$\text{PSI} \ge 0.25$**: Significant drift; trigger automated retraining workflow.

### 14.2 Production Infrastructure Scope
- **Current Scale**: Single/multi-instance containerized microservices managed via Docker / Docker Compose or container platforms (AWS ECS, Google Cloud Run, Azure Container Apps).
- **Future Scale**: For institutional banks processing hundreds of applications per second with strict data governance, Kubernetes (EKS/GKE), an enterprise Feature Store (Feast), and cloud data warehouses (BigQuery/Snowflake) can be integrated without modifying the core Scikit-Learn pipeline.

---

## 15. License & Contributions

- **License**: MIT License.
- **Dataset**: Kaggle "Give Me Some Credit" terms and conditions.