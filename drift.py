import pandas as pd
import numpy as np
from scipy import stats


FEATURES = [
    "age", "tenure_months", "monthly_charges",
    "num_products", "support_calls", "satisfaction_score"
]


def detect_drift(reference_df, current_df, threshold = 0.05):
    """
    Detect data drift between reference (training) and current (production) data.
    Uses Kolmogorov-Smirnov test for numerical features.
    """
    results = {}
    drift_detected = False

    for feature in FEATURES:
        ref = reference_df[feature].dropna()
        cur = current_df[feature].dropna()

        # KS Test
        ks_stat, p_value = stats.ks_2samp(ref, cur)

        drifted = p_value < threshold

        if drifted:
            drift_detected = True

        results[feature] = {
            "ks_statistic": round(ks_stat, 4),
            "p_value": round(p_value, 4),
            "drifted": drifted,
            "reference_mean": round(ref.mean(), 2),
            "current_mean": round(cur.mean(), 2),
            "mean_shift": round(cur.mean() - ref.mean(), 2),
            "reference_std": round(ref.std(), 2),
            "current_std": round(cur.std(), 2)
        }

    # Overall drift score
    n_drifted = sum(1 for f in results.values() if f["drifted"])
    drift_score = round(n_drifted / len(FEATURES) * 100, 1)

    return {
        "drift_detected": drift_detected,
        "drift_score": drift_score,
        "n_features_drifted": n_drifted,
        "total_features": len(FEATURES),
        "feature_results": results,
        "recommendation": get_recommendation(drift_score)
    }


def get_recommendation(drift_score):
    if drift_score == 0:
        return "✅ No drift detected. Model is healthy and no action needed."
    elif drift_score <= 33:
        return "⚠️ Minor drift detected. Monitor closely but retraining not urgent."
    elif drift_score <= 66:
        return "🔶 Moderate drift detected. Schedule model retraining soon."
    else:
        return "🚨 Severe drift detected! Immediate retraining recommended."


def get_drift_summary(drift_results):
    """Return drift results as a clean DataFrame."""
    rows = []
    for feature, result in drift_results["feature_results"].items():
        rows.append({
            "Feature": feature,
            "Drifted": "🔴 Yes" if result["drifted"] else "🟢 No",
            "KS Statistic": result["ks_statistic"],
            "P-Value": result["p_value"],
            "Reference Mean": result["reference_mean"],
            "Current Mean": result["current_mean"],
            "Mean Shift": result["mean_shift"]
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from data.generate_data import generate_churn_data

    print("=== Testing with NO drift ===")
    reference = generate_churn_data(n_samples=1000, random_seed=42)
    current_normal = generate_churn_data(n_samples=500, random_seed=99)
    results = detect_drift(reference, current_normal)
    print(f"Drift Detected: {results['drift_detected']}")
    print(f"Drift Score: {results['drift_score']}%")
    print(f"Recommendation: {results['recommendation']}")

    print("\n=== Testing WITH drift ===")
    current_drifted = generate_churn_data(n_samples=500, drift=True)
    results_drifted = detect_drift(reference, current_drifted)
    print(f"Drift Detected: {results_drifted['drift_detected']}")
    print(f"Drift Score: {results_drifted['drift_score']}%")
    print(f"Recommendation: {results_drifted['recommendation']}")