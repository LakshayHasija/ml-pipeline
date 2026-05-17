# 🤖 ML Pipeline with Drift Monitoring

An end-to-end machine learning pipeline for customer churn prediction with real-time data drift detection, REST API deployment, and experiment tracking.

---

## 🚀 What It Does

> "Is our model still reliable — or has the real world changed?"

This pipeline trains a churn prediction model, deploys it as a REST API, monitors incoming data for drift, and tracks all experiments — exactly how production ML systems work.

---

## 🧠 How It Works

```
Raw Data Generation
        ↓
Model Training (Random Forest)
        ↓
FastAPI REST API Deployment
        ↓
MLflow Experiment Tracking
        ↓
KS-Test Drift Detection
        ↓
Retraining Alert + One-Click Retrain
        ↓
Streamlit Monitoring Dashboard
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | Scikit-learn (Random Forest) |
| API | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Drift Detection | SciPy (KS-Test) |
| Frontend | Streamlit |
| Language | Python 3.x |

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| ROC-AUC | 0.785 |
| Recall | 0.753 |
| F1 Score | 0.498 |
| Accuracy | 0.663 |

> High recall is prioritized — better to flag false alarms than miss real churners.

---

## 📡 Drift Detection

Uses the **Kolmogorov-Smirnov (KS) Test** to compare feature distributions between training and production data.

| Drift Score | Status |
|---|---|
| 0% | ✅ No drift — model healthy |
| 1–33% | ⚠️ Minor drift — monitor closely |
| 34–66% | 🔶 Moderate drift — retrain soon |
| 67–100% | 🚨 Severe drift — retrain immediately |

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Model health check |
| POST | `/predict` | Predict churn for a customer |

### Example Request
```json
POST /predict
{
  "age": 55,
  "tenure_months": 3,
  "monthly_charges": 95,
  "num_products": 1,
  "support_calls": 6,
  "satisfaction_score": 1
}
```

### Example Response
```json
{
  "churn_probability": 0.8136,
  "churn_prediction": true,
  "risk_level": "🔴 High Risk",
  "message": "Customer is very likely to churn. Immediate intervention recommended."
}
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/ml-pipeline.git
cd ml-pipeline
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python train.py
```

### 5. Run the API
```bash
python api.py
```

### 6. Run the dashboard
```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ml-pipeline/
├── app.py                  # Streamlit monitoring dashboard
├── api.py                  # FastAPI REST API
├── train.py                # Model training + evaluation
├── drift.py                # KS-test drift detection
├── mlflow_tracker.py       # MLflow experiment logging
├── data/
│   └── generate_data.py    # Synthetic churn dataset generator
├── models/                 # Saved model + scaler
├── requirements.txt
└── .gitignore
```

---

## 🔮 Future Improvements

- Connect to real customer database
- Add email alerts when drift is detected
- Docker containerization for deployment
- Support for multiple model types (XGBoost, LightGBM)
- Automated retraining scheduler with Airflow

---
Built with Scikit-learn, FastAPI, MLflow, SciPy & Streamlit