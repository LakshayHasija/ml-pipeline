import pandas as pd
import numpy as np


def generate_churn_data(n_samples: int = 1000, random_seed: int = 42, drift: bool = False) -> pd.DataFrame:
    """
    Generate synthetic customer churn dataset.
    If drift=True, simulate data drift by shifting feature distributions.
    """
    np.random.seed(random_seed)

    if not drift:
        # Normal production data
        age = np.random.normal(40, 12, n_samples).clip(18, 80)
        tenure_months = np.random.normal(24, 12, n_samples).clip(1, 72)
        monthly_charges = np.random.normal(65, 20, n_samples).clip(20, 120)
        num_products = np.random.randint(1, 5, n_samples)
        support_calls = np.random.poisson(2, n_samples)
        satisfaction_score = np.random.randint(1, 6, n_samples)
    else:
        # Drifted data - distributions have shifted
        age = np.random.normal(55, 15, n_samples).clip(18, 80)          # older customers
        tenure_months = np.random.normal(12, 8, n_samples).clip(1, 72)  # shorter tenure
        monthly_charges = np.random.normal(85, 25, n_samples).clip(20, 120)  # higher charges
        num_products = np.random.randint(1, 3, n_samples)                # fewer products
        support_calls = np.random.poisson(4, n_samples)                  # more support calls
        satisfaction_score = np.random.randint(1, 4, n_samples)          # lower satisfaction

    # Churn probability based on features
    churn_prob = (
        0.3 * (support_calls > 3).astype(float) +
        0.25 * (satisfaction_score < 3).astype(float) +
        0.2 * (tenure_months < 12).astype(float) +
        0.15 * (monthly_charges > 90).astype(float) +
        0.1 * (num_products == 1).astype(float)
    ).clip(0, 1)

    churn = np.random.binomial(1, churn_prob, n_samples)

    df = pd.DataFrame({
        "age": age.round(1),
        "tenure_months": tenure_months.round(1),
        "monthly_charges": monthly_charges.round(2),
        "num_products": num_products,
        "support_calls": support_calls,
        "satisfaction_score": satisfaction_score,
        "churn": churn
    })

    return df


if __name__ == "__main__":
    df = generate_churn_data()
    print(df.head(10))
    print(f"\nShape: {df.shape}")
    print(f"Churn Rate: {df['churn'].mean():.2%}")
    df.to_csv("data/churn_data.csv", index=False)
    print("✅ Dataset saved to data/churn_data.csv")