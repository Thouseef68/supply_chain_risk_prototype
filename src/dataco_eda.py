import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"
OUTPUT_DIR = "outputs"

TARGET = "Late_delivery_risk"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DATACO SUPPLY CHAIN - EXPLORATORY DATA ANALYSIS")
print("=" * 80)

df = pd.read_csv(DATA_PATH, encoding="latin1")

print("\nDataset:")
print(f"Rows    : {len(df):,}")
print(f"Columns : {len(df.columns)}")


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

print("\n[1] TARGET DISTRIBUTION")
print("-" * 60)

target_counts = df[TARGET].value_counts()
target_percent = df[TARGET].value_counts(normalize=True) * 100

print(target_counts)

print("\nPercentages:")
for value in target_counts.index:
    print(
        f"{value}: "
        f"{target_counts[value]:,} "
        f"({target_percent[value]:.2f}%)"
    )


# ============================================================
# TARGET BY CATEGORICAL FEATURES
# ============================================================

categorical_columns = [
    "Type",
    "Category Name",
    "Customer Country",
    "Customer Segment",
    "Customer State",
    "Department Name",
    "Market",
    "Order Country",
    "Order Region",
    "Order State",
    "Product Name",
    "Shipping Mode"
]


print("\n[2] DISRUPTION/LATE RISK BY CATEGORICAL FEATURES")
print("-" * 60)

category_results = []

for col in categorical_columns:

    if col not in df.columns:
        continue

    grouped = (
        df.groupby(col)[TARGET]
        .agg(["count", "mean"])
        .sort_values("mean", ascending=False)
    )

    grouped["risk_percent"] = grouped["mean"] * 100

    print(f"\n### {col}")

    print(
        grouped.head(15).to_string()
    )

    temp = grouped.reset_index()
    temp["feature"] = col

    category_results.append(temp)


if category_results:

    categorical_summary = pd.concat(
        category_results,
        ignore_index=True
    )

    categorical_summary.to_csv(
        f"{OUTPUT_DIR}/categorical_risk_analysis.csv",
        index=False
    )


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numeric_columns = [
    "Days for shipment (scheduled)",
    "Benefit per order",
    "Sales per customer",
    "Latitude",
    "Longitude",
    "Order Item Discount",
    "Order Item Discount Rate",
    "Order Item Product Price",
    "Order Item Profit Ratio",
    "Order Item Quantity",
    "Sales",
    "Order Item Total",
    "Order Profit Per Order",
    "Product Price"
]


print("\n[3] NUMERICAL FEATURE ANALYSIS")
print("-" * 60)

numeric_results = []

for col in numeric_columns:

    if col not in df.columns:
        continue

    correlation = df[col].corr(df[TARGET])

    numeric_results.append({
        "feature": col,
        "correlation": correlation,
        "abs_correlation": abs(correlation)
    })

numeric_summary = pd.DataFrame(
    numeric_results
).sort_values(
    "abs_correlation",
    ascending=False
)

print(
    numeric_summary.to_string(index=False)
)

numeric_summary.to_csv(
    f"{OUTPUT_DIR}/numeric_correlations.csv",
    index=False
)


# ============================================================
# RISK BY SHIPPING MODE
# ============================================================

print("\n[4] SHIPPING MODE RISK")
print("-" * 60)

shipping_risk = (
    df.groupby("Shipping Mode")[TARGET]
    .agg(["count", "sum", "mean"])
)

shipping_risk["risk_percent"] = (
    shipping_risk["mean"] * 100
)

print(
    shipping_risk.sort_values(
        "risk_percent",
        ascending=False
    ).to_string()
)

shipping_risk.to_csv(
    f"{OUTPUT_DIR}/shipping_mode_risk.csv"
)


# ============================================================
# RISK BY SCHEDULED SHIPPING DAYS
# ============================================================

print("\n[5] SCHEDULED SHIPPING DAYS")
print("-" * 60)

scheduled_risk = (
    df.groupby(
        "Days for shipment (scheduled)"
    )[TARGET]
    .agg(["count", "sum", "mean"])
)

scheduled_risk["risk_percent"] = (
    scheduled_risk["mean"] * 100
)

print(
    scheduled_risk.to_string()
)

scheduled_risk.to_csv(
    f"{OUTPUT_DIR}/scheduled_days_risk.csv"
)


# ============================================================
# CORRELATION MATRIX
# ============================================================

print("\n[6] CORRELATION ANALYSIS")
print("-" * 60)

numeric_df = df.select_dtypes(
    include=["int64", "float64"]
)

correlation_matrix = numeric_df.corr()

target_correlation = (
    correlation_matrix[TARGET]
    .drop(TARGET)
    .sort_values(
        key=abs,
        ascending=False
    )
)

print("\nCorrelation with target:")

print(
    target_correlation.to_string()
)

target_correlation.to_csv(
    f"{OUTPUT_DIR}/target_correlations.csv"
)


# ============================================================
# HIGH-CORRELATION FEATURE CHECK
# ============================================================

print("\n[7] POTENTIALLY STRONG FEATURES")
print("-" * 60)

for feature, correlation in target_correlation.items():

    if abs(correlation) >= 0.10:

        print(
            f"{feature:<40} "
            f"{correlation:+.4f}"
        )


# ============================================================
# DATE ANALYSIS
# ============================================================

print("\n[8] TEMPORAL ANALYSIS")
print("-" * 60)

df["order_date"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["year"] = df["order_date"].dt.year
df["month"] = df["order_date"].dt.month
df["day_of_week"] = df["order_date"].dt.dayofweek

for col in ["year", "month", "day_of_week"]:

    result = (
        df.groupby(col)[TARGET]
        .agg(["count", "mean"])
    )

    result["risk_percent"] = (
        result["mean"] * 100
    )

    print(f"\n{col}")

    print(
        result.to_string()
    )


# ============================================================
# HIGH-RISK COMBINATIONS
# ============================================================

print("\n[9] SHIPPING MODE + MARKET")
print("-" * 60)

combo = (
    df.groupby(
        ["Shipping Mode", "Market"]
    )[TARGET]
    .agg(["count", "mean"])
)

combo["risk_percent"] = combo["mean"] * 100

combo = combo.sort_values(
    "risk_percent",
    ascending=False
)

print(
    combo.head(20).to_string()
)

combo.to_csv(
    f"{OUTPUT_DIR}/shipping_market_risk.csv"
)


# ============================================================
# DATA QUALITY
# ============================================================

print("\n[10] DATA QUALITY")
print("-" * 60)

print(
    "Missing values:",
    df.isnull().sum().sum()
)

print(
    "Duplicate rows:",
    df.duplicated().sum()
)

print(
    "Target values:",
    sorted(df[TARGET].unique())
)


# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "rows": len(df),
    "columns": len(df.columns),
    "target_0": int((df[TARGET] == 0).sum()),
    "target_1": int((df[TARGET] == 1).sum()),
    "target_0_percent": float(
        (df[TARGET] == 0).mean() * 100
    ),
    "target_1_percent": float(
        (df[TARGET] == 1).mean() * 100
    ),
    "missing_values": int(
        df.isnull().sum().sum()
    ),
    "duplicate_rows": int(
        df.duplicated().sum()
    )
}

pd.DataFrame(
    [summary]
).to_csv(
    f"{OUTPUT_DIR}/dataco_eda_summary.csv",
    index=False
)


print("\n" + "=" * 80)
print("EDA COMPLETE")
print("=" * 80)

print("\nGenerated files:")

print("outputs/categorical_risk_analysis.csv")
print("outputs/numeric_correlations.csv")
print("outputs/shipping_mode_risk.csv")
print("outputs/scheduled_days_risk.csv")
print("outputs/target_correlations.csv")
print("outputs/shipping_market_risk.csv")
print("outputs/dataco_eda_summary.csv")