# Enterprise Credit Risk Scoring API

An end-to-end Machine Learning microservice that predicts the probability of loan default. This project demonstrates the complete transition from exploratory data analysis (EDA) in Jupyter Notebooks to a production-ready, containerized REST API.

## 🚀 Project Highlights
- **Robust Feature Engineering:** Utilizes custom Scikit-Learn `BaseEstimator` and `TransformerMixin` classes to prevent data leakage and ensure training-serving symmetry.
- **Experiment Tracking:** Uses **MLflow** to track hyperparameter tuning, metrics (ROC-AUC, Precision, Recall), and model artifacts.
- **Interpretability:** Leverages **SHAP** (SHapley Additive exPlanations) to provide feature importance and satisfy financial regulatory explainability requirements.
- **Optimized Business Logic:** Calculates a custom probability threshold by maximizing the F1 score along the Precision-Recall curve to balance revenue and risk.
- **Enterprise API:** Built with **FastAPI** and **Pydantic** for asynchronous inference and strict data payload validation.
- **DevOps Ready:** Fully containerized with **Docker** (Linux-compatible) and tested via **Pytest**.

## 🛠️ Tech Stack
* **Machine Learning:** Scikit-Learn, XGBoost, SHAP
* **MLOps & Tracking:** MLflow, Joblib
* **API & Backend:** FastAPI, Pydantic, Uvicorn
* **DevOps:** Docker, Pytest
* **Data Processing:** Pandas, NumPy

## 📂 Project Structure

    credit-risk-ml/
    ├── api/
    │   ├── __init__.py
    │   ├── main.py              # FastAPI application and lifespan events
    │   └── schemas.py           # Pydantic models for payload validation
    ├── artifacts/               # Serialized models and optimal threshold
    ├── data/                    # Raw Kaggle dataset (cs-training.csv)
    ├── notebooks/               # Jupyter notebooks for EDA, preprocessing, and training
    ├── src/
    │   ├── __init__.py
    │   └── features.py          # Custom scikit-learn transformers
    ├── tests/
    │   ├── __init__.py
    │   └── test_api.py          # Pytest automated test suite
    ├── Dockerfile               # Linux container blueprint
    ├── .dockerignore
    ├── requirements.txt         # Strictly pinned dependency lockfile
    └── README.md


## 🐳 Running with Docker (Recommended)
This API is packaged into an isolated Linux container, ensuring it runs identically on any machine or cloud provider.

1. **Build the Image:**
    ```bash
    docker build -t credit-risk-api:latest .
    ```

2. **Run the Container:**
    ```bash
    docker run -d -p 8080:8000 --name credit-api credit-risk-api:latest
    ```

3. **Access the API Documentation:**
   Open your browser and navigate to the interactive Swagger UI:
   [http://localhost:8080/docs](http://localhost:8080/docs)

## 💻 Running Locally (Development Mode)
If you wish to run the project without Docker:

1. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use: venv\Scripts\activate
    ```

2. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Start the FastAPI server:**
    ```bash
    uvicorn api.main:app --reload
    ```

## 🧪 Automated Testing
To ensure the integrity of the data validation and the ML pipeline, run the Pytest suite:
```bash
python -m pytest


# API USage example
{
  "RevolvingUtilizationOfUnsecuredLines": 0.5,
  "age": 45,
  "NumberOfTime30-59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.3,
  "MonthlyIncome": 6000.0,
  "NumberOfOpenCreditLinesAndLoans": 5,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60-89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 2
}