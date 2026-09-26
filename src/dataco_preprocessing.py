import pandas as pd
import numpy as np
import os

# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"
OUTPUT_PATH = "outputs/dataco_cleaned.csv"

TARGET = "Late_delivery_risk"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("DATACO SUPPLY CHAIN PREPROCESSING")
print("=" * 80)

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH, encoding="latin1")

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# TARGET CHECK
# ============================================================

print("\n[2] TARGET ANALYSIS")
print("-" * 50)

print(df[TARGET].value_counts())

print("\nTarget percentage:")
print(
    (df[TARGET].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# DATE FEATURES
# ============================================================

print("\n[3] CREATING DATE FEATURES")
print("-" * 50)

df["order_date"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_day"] = df["order_date"].dt.day
df["order_dayofweek"] = df["order_date"].dt.dayofweek
df["order_week"] = df["order_date"].dt.isocalendar().week.astype(float)

print("Date features created.")


# ============================================================
# LEAKAGE / POST-OUTCOME COLUMNS
# ============================================================

print("\n[4] REMOVING TARGET-LEAKAGE COLUMNS")
print("-" * 50)

leakage_columns = [

    # Directly reveals delivery outcome
    "Delivery Status",

    # Actual shipping duration is known after shipment
    "Days for shipping (real)",

    # Actual shipping date is known after shipment
    "shipping date (DateOrders)",

    # Order status can reflect events after the prediction point
    "Order Status",

]


# ============================================================
# PERSONAL / SENSITIVE / USELESS COLUMNS
# ============================================================

print("\n[5] REMOVING PERSONAL / IDENTIFIER COLUMNS")
print("-" * 50)

remove_columns = [

    # Customer personal information
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",

    # Customer identifiers
    "Customer Id",
    "Order Customer Id",

    # Order identifiers
    "Order Id",
    "Order Item Id",

    # Product/order identifiers
    "Product Card Id",
    "Order Item Cardprod Id",
    "Category Id",
    "Product Category Id",
    "Department Id",

    # High-cardinality / unnecessary identifiers
    "Order Zipcode",

    # Product metadata not useful for prediction
    "Product Description",
    "Product Image",

    # Constant column
    "Product Status",

    # Original date string
    "order date (DateOrders)",

]


# ============================================================
# COMBINE DROP LIST
# ============================================================

drop_columns = list(
    dict.fromkeys(
        leakage_columns + remove_columns
    )
)

existing_drop_columns = [
    col for col in drop_columns
    if col in df.columns
]

print(f"Columns to remove: {len(existing_drop_columns)}")

for col in existing_drop_columns:
    print("  -", col)

df = df.drop(
    columns=existing_drop_columns,
    errors="ignore"
)


# ============================================================
# REMOVE TEMPORARY DATE COLUMN
# ============================================================

df = df.drop(
    columns=["order_date"],
    errors="ignore"
)


# ============================================================
# MISSING VALUE HANDLING
# ============================================================

print("\n[6] HANDLING MISSING VALUES")
print("-" * 50)

numeric_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for col in numeric_columns:

    if col != TARGET:

        df[col] = df[col].fillna(
            df[col].median()
        )


for col in categorical_columns:

    df[col] = df[col].fillna(
        "Unknown"
    )


print(
    "Remaining missing values:",
    df.isnull().sum().sum()
)


# ============================================================
# TARGET VALIDATION
# ============================================================

print("\n[7] TARGET VALIDATION")
print("-" * 50)

df = df[
    df[TARGET].isin([0, 1])
].copy()

df[TARGET] = df[TARGET].astype(int)

print("Target values:")
print(df[TARGET].value_counts())


# ============================================================
# FINAL DATASET INFORMATION
# ============================================================

print("\n[8] FINAL DATASET")
print("-" * 50)

print("Rows    :", f"{df.shape[0]:,}")
print("Columns :", df.shape[1])

print("\nFinal columns:")

for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")


# ============================================================
# NUMERICAL / CATEGORICAL SUMMARY
# ============================================================

print("\n[9] FEATURE TYPES")
print("-" * 50)

numeric_features = [
    col for col in df.select_dtypes(
        include=["int64", "float64"]
    ).columns
    if col != TARGET
]

categorical_features = list(
    df.select_dtypes(
        include=["object"]
    ).columns
)

print(
    "Numerical features  :",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    "outputs",
    exist_ok=True
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\n[10] SAVED")
print("-" * 50)

print(
    f"Clean dataset saved to:\n{OUTPUT_PATH}"
)

print("\n" + "=" * 80)
print("PREPROCESSING COMPLETE")
print("=" * 80)