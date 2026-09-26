import pandas as pd
import glob
import matplotlib.pyplot as plt

# Load dataset
file = glob.glob("data/*.csv")[0]
df = pd.read_csv(file)

print("=" * 60)
print("SUPPLY CHAIN DATA ANALYSIS")
print("=" * 60)

# --------------------------------------------------
# 1. Dataset information
# --------------------------------------------------

print("\nDataset Shape:")
print(df.shape)

print("\nColumns:")
for column in df.columns:
    print("-", column)

# --------------------------------------------------
# 2. Target distribution
# --------------------------------------------------

print("\nDisruption Distribution:")
print(df["Disruption_Occurred"].value_counts())

print("\nDisruption Percentage:")
print(
    df["Disruption_Occurred"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

# --------------------------------------------------
# 3. Numerical statistics
# --------------------------------------------------

print("\nNumerical Statistics:")
print(df.describe())

# --------------------------------------------------
# 4. Categorical values
# --------------------------------------------------

categorical_columns = [
    "Origin_Port",
    "Destination_Port",
    "Transport_Mode",
    "Product_Category",
    "Weather_Condition"
]

print("\nCategorical Information:")

for column in categorical_columns:
    print(f"\n{column}")
    print(df[column].value_counts())

# --------------------------------------------------
# 5. Disruption rate by categories
# --------------------------------------------------

for column in categorical_columns:

    print(f"\nDisruption Rate by {column}:")

    result = (
        df.groupby(column)["Disruption_Occurred"]
        .mean()
        .sort_values(ascending=False)
    )

    print((result * 100).round(2))

# --------------------------------------------------
# 6. Correlation
# --------------------------------------------------

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

print("\nCorrelation with Disruption:")

correlation = (
    df[numeric_columns]
    .corr()["Disruption_Occurred"]
    .sort_values(ascending=False)
)

print(correlation)

# --------------------------------------------------
# 7. Create output directory
# --------------------------------------------------

import os
os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------
# 8. Target distribution chart
# --------------------------------------------------

plt.figure(figsize=(7, 5))

df["Disruption_Occurred"].value_counts().sort_index().plot(
    kind="bar"
)

plt.title("Supply Chain Disruption Distribution")
plt.xlabel("Disruption Occurred")
plt.ylabel("Number of Shipments")

plt.xticks(
    [0, 1],
    ["No Disruption", "Disruption"],
    rotation=0
)

plt.tight_layout()

plt.savefig(
    "outputs/disruption_distribution.png",
    dpi=300
)

plt.show()

print("\nEDA completed successfully.")