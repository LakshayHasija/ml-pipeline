import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler
from data.generate_data import generate_churn_data


MODEL_PATH = "models/churn_model.pkl"
SCALER_PATH = "models/scaler.pkl"
FEATURES = [
    "age", "tenure_months", "monthly_charges",
    "num_products", "support_calls", "satisfaction_score"
]


def train(save: bool = True) -> dict:
    """Train a Random Forest model on churn data."""

    # Generate + load data
    df = generate_churn_data(n_samples=2000)
    X = df[FEATURES]
    y = df["churn"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        min_samples_split=10,
        random_state=42,
        class_weight="balanced"
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }

    # Feature importance
    feature_importance = dict(zip(FEATURES, model.feature_importances_.round(4)))

    # Save model and scaler
    if save:
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)
        print("✅ Model saved to models/churn_model.pkl")
        print("✅ Scaler saved to models/scaler.pkl")

    print("\n📊 Model Performance:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"  {k}: {v}")

    print("\n🌲 Feature Importance:")
    for k, v in sorted(feature_importance.items(), key=lambda x: x[1], reverse=True):
        print(f"  {k}: {v}")

    return {
        "metrics": metrics,
        "feature_importance": feature_importance,
        "model": model,
        "scaler": scaler
    }


if __name__ == "__main__":
    train()