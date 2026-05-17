from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import os

app = FastAPI(
    title="Churn Prediction API",
    description="ML-powered customer churn prediction with drift monitoring",
    version="1.0.0"
)

# Load model and scaler
MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = None
scaler = None

def load_model():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return True
    return False

load_model()


class CustomerFeatures(BaseModel):
    age: float = Field(..., ge=18, le=80, description="Customer age")
    tenure_months: float = Field(..., ge=1, le=72, description="Months as customer")
    monthly_charges: float = Field(..., ge=20, le=120, description="Monthly bill amount")
    num_products: int = Field(..., ge=1, le=4, description="Number of products subscribed")
    support_calls: int = Field(..., ge=0, description="Number of support calls made")
    satisfaction_score: int = Field(..., ge=1, le=5, description="Customer satisfaction score")


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    message: str


@app.get("/")
def root():
    return {
        "message": "Churn Prediction API is running!",
        "endpoints": ["/predict", "/health", "/docs"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if model is None or scaler is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    features = np.array([[
        customer.age,
        customer.tenure_months,
        customer.monthly_charges,
        customer.num_products,
        customer.support_calls,
        customer.satisfaction_score
    ]])

    features_scaled = scaler.transform(features)
    churn_prob = model.predict_proba(features_scaled)[0][1]
    churn_pred = bool(churn_prob >= 0.5)

    if churn_prob >= 0.7:
        risk_level = "🔴 High Risk"
        message = "Customer is very likely to churn. Immediate intervention recommended."
    elif churn_prob >= 0.4:
        risk_level = "🟡 Medium Risk"
        message = "Customer shows churn signals. Consider proactive outreach."
    else:
        risk_level = "🟢 Low Risk"
        message = "Customer is unlikely to churn. Continue normal engagement."

    return PredictionResponse(
        churn_probability=round(churn_prob, 4),
        churn_prediction=churn_pred,
        risk_level=risk_level,
        message=message
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)