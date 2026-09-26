import pandas as pd
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from preprocessing import load_data, prepare_features


# --------------------------------------------------
# Load data
# --------------------------------------------------

df = load_data()

X, y = prepare_features(df)


# --------------------------------------------------
# Scale features
# --------------------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# --------------------------------------------------
# Test different numbers of clusters
# --------------------------------------------------

print("=" * 60)
print("K-MEANS SUPPLY CHAIN RISK CLUSTERING")
print("=" * 60)

results = []

for k in range(2, 6):

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    results.append((k, score))

    print(
        f"Clusters: {k} | "
        f"Silhouette Score: {score:.4f}"
    )


# --------------------------------------------------
# Select best cluster count
# --------------------------------------------------

best_k, best_score = max(
    results,
    key=lambda x: x[1]
)

print("\nSelected number of clusters:", best_k)
print("Best silhouette score:", round(best_score, 4))


# --------------------------------------------------
# Train final clustering model
# --------------------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=10
)

clusters = final_model.fit_predict(X_scaled)


# --------------------------------------------------
# Add clusters to original data
# --------------------------------------------------

df["Risk_Cluster"] = clusters


# --------------------------------------------------
# Analyze clusters
# --------------------------------------------------

print("\nCluster Distribution:")
print(df["Risk_Cluster"].value_counts().sort_index())

print("\nDisruption Rate by Cluster:")

cluster_risk = (
    df.groupby("Risk_Cluster")["Disruption_Occurred"]
    .agg(["count", "mean"])
)

cluster_risk["Disruption_Rate_%"] = (
    cluster_risk["mean"] * 100
).round(2)

print(cluster_risk)


# --------------------------------------------------
# Save result
# --------------------------------------------------

os.makedirs("outputs", exist_ok=True)

df.to_csv(
    "outputs/clustered_shipments.csv",
    index=False
)

print("\nSaved:")
print("outputs/clustered_shipments.csv")