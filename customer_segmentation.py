"""
Customer Segmentation using K-Means clustering
Sample demonstration project for the Thiranex Data Analytics internship.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def load_data(path="customer_data.csv"):
    return pd.read_csv(path)


def main():
    df = load_data()

    required = ["Age", "Annual_Income", "Spending_Score"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    features = df[required].copy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    # K-Means groups customers based on similarity in the selected features.
    model = KMeans(n_clusters=4, random_state=42, n_init=10)
    df["Customer_Segment"] = model.fit_predict(scaled)

    score = silhouette_score(scaled, df["Customer_Segment"])
    df.to_csv("segmented_customers.csv", index=False)

    print("Customer Segmentation completed.")
    print(f"Number of customers: {len(df)}")
    print(f"Number of segments: {df['Customer_Segment'].nunique()}")
    print(f"Silhouette score: {score:.3f}")
    print("\nSample segmented customers:")
    print(df.head())

    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(
        df["Annual_Income"],
        df["Spending_Score"],
        c=df["Customer_Segment"],
        s=60
    )
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.title("Customer Segments")
    plt.colorbar(scatter, label="Segment")
    plt.tight_layout()
    plt.savefig("customer_segments.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    main()
