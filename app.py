import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import joblib
import subprocess
import sys
from data.generate_data import generate_churn_data
from train import train, FEATURES
from drift import detect_drift, get_drift_summary
from mlflow_tracker import log_training_run, get_all_runs

st.set_page_config(
    page_title="ML Pipeline Monitor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 ML Pipeline with Drift Monitoring")
st.markdown("End-to-end churn prediction pipeline with real-time drift detection and experiment tracking.")
st.divider()

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("models/churn_model.pkl")
        scaler = joblib.load("models/scaler.pkl")
        return model, scaler
    except:
        return None, None

model, scaler = load_model()

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict", "📡 Drift Monitor", "📊 Model Performance", "🧪 Experiment History"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("🔮 Customer Churn Prediction")
    st.markdown("Enter customer details to predict churn probability.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 80, 40)
        tenure_months = st.slider("Tenure (Months)", 1, 72, 24)
        monthly_charges = st.slider("Monthly Charges (₹)", 20, 120, 65)

    with col2:
        num_products = st.selectbox("Number of Products", [1, 2, 3, 4])
        support_calls = st.slider("Support Calls", 0, 10, 2)
        satisfaction_score = st.selectbox("Satisfaction Score", [1, 2, 3, 4, 5], index=2)

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        if model is None:
            st.error("Model not loaded. Please train the model first in the Model Performance tab.")
        else:
            features = np.array([[age, tenure_months, monthly_charges,
                                   num_products, support_calls, satisfaction_score]])
            features_scaled = scaler.transform(features)
            churn_prob = model.predict_proba(features_scaled)[0][1]

            st.divider()

            col1, col2, col3 = st.columns(3)
            col1.metric("Churn Probability", f"{churn_prob:.1%}")
            col2.metric("Prediction", "Will Churn 🔴" if churn_prob >= 0.5 else "Will Stay 🟢")

            if churn_prob >= 0.7:
                risk = "🔴 High Risk"
                color = "error"
            elif churn_prob >= 0.4:
                risk = "🟡 Medium Risk"
                color = "warning"
            else:
                risk = "🟢 Low Risk"
                color = "success"

            col3.metric("Risk Level", risk)

            # Probability gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=churn_prob * 100,
                title={"text": "Churn Probability (%)"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "darkred" if churn_prob >= 0.7 else "orange" if churn_prob >= 0.4 else "green"},
                    "steps": [
                        {"range": [0, 40], "color": "#d4edda"},
                        {"range": [40, 70], "color": "#fff3cd"},
                        {"range": [70, 100], "color": "#f8d7da"}
                    ]
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — DRIFT MONITOR
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("📡 Data Drift Monitor")

    col1, col2 = st.columns(2)
    with col1:
        n_current = st.slider("Current Data Sample Size", 100, 2000, 500)
    with col2:
        simulate_drift = st.checkbox("🔀 Simulate Data Drift", value=False)

    if st.button("🔍 Run Drift Detection", type="primary", use_container_width=True):
        reference_df = generate_churn_data(n_samples=2000, random_seed=42)
        current_df = generate_churn_data(
            n_samples=n_current,
            random_seed=99,
            drift=simulate_drift
        )

        drift_results = detect_drift(reference_df, current_df)
        summary_df = get_drift_summary(drift_results)

        # Overall status
        if drift_results["drift_detected"]:
            st.error(f"🚨 Drift Detected! {drift_results['recommendation']}")
        else:
            st.success(drift_results["recommendation"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Drift Score", f"{drift_results['drift_score']}%")
        col2.metric("Features Drifted", f"{drift_results['n_features_drifted']}/{drift_results['total_features']}")
        col3.metric("Status", "⚠️ Drifted" if drift_results["drift_detected"] else "✅ Healthy")

        st.divider()
        st.markdown("### Feature-Level Drift Analysis")
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        # Distribution comparison charts
        st.markdown("### Distribution Comparison")
        feature_to_plot = st.selectbox("Select Feature", FEATURES)

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=reference_df[feature_to_plot],
            name="Reference (Training)",
            opacity=0.6,
            marker_color="#636EFA",
            nbinsx=30
        ))
        fig.add_trace(go.Histogram(
            x=current_df[feature_to_plot],
            name="Current (Production)",
            opacity=0.6,
            marker_color="#EF553B",
            nbinsx=30
        ))
        fig.update_layout(
            barmode="overlay",
            title=f"Distribution Comparison: {feature_to_plot}",
            xaxis_title=feature_to_plot,
            yaxis_title="Count",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Model Performance & Retraining")

    if model is None:
        st.warning("No model found. Train one below.")
    else:
        st.success("✅ Model loaded and ready")

    if st.button("🔄 Retrain Model", type="primary", use_container_width=True):
        with st.spinner("Training model..."):
            result = train(save=True)
            log_training_run(
                metrics=result["metrics"],
                feature_importance=result["feature_importance"],
                model=result["model"],
                scaler=result["scaler"]
            )
            st.cache_resource.clear()
            model, scaler = load_model()

        st.success("✅ Model retrained and logged to MLflow!")

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Accuracy", result["metrics"]["accuracy"])
        col2.metric("Precision", result["metrics"]["precision"])
        col3.metric("Recall", result["metrics"]["recall"])
        col4.metric("F1 Score", result["metrics"]["f1_score"])
        col5.metric("ROC AUC", result["metrics"]["roc_auc"])

        st.divider()

        # Feature importance chart
        st.markdown("### 🌲 Feature Importance")
        fi_df = pd.DataFrame(
            result["feature_importance"].items(),
            columns=["Feature", "Importance"]
        ).sort_values("Importance", ascending=True)

        fig = px.bar(
            fi_df, x="Importance", y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale="Blues",
            title="Feature Importance (Random Forest)"
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — EXPERIMENT HISTORY
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("🧪 MLflow Experiment History")

    runs = get_all_runs()

    if not runs:
        st.info("No runs logged yet. Train a model in the Model Performance tab.")
    else:
        st.success(f"Found {len(runs)} experiment run(s)")

        runs_df = pd.DataFrame(runs)
        runs_df["start_time"] = pd.to_datetime(runs_df["start_time"], unit="ms")

        st.dataframe(
            runs_df[["run_id", "start_time", "accuracy", "precision", "recall", "f1_score", "roc_auc", "status"]],
            use_container_width=True,
            hide_index=True
        )

        # Metrics over time
        if len(runs) > 1:
            st.markdown("### Metrics Over Time")
            fig = go.Figure()
            for metric in ["accuracy", "f1_score", "roc_auc"]:
                fig.add_trace(go.Scatter(
                    x=runs_df["start_time"],
                    y=runs_df[metric],
                    name=metric,
                    mode="lines+markers"
                ))
            fig.update_layout(
                title="Model Metrics Across Runs",
                xaxis_title="Time",
                yaxis_title="Score",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built with Scikit-learn, FastAPI, MLflow, Evidently & Streamlit | ML Pipeline with Drift Monitoring")