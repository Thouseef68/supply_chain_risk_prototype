import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"

MODEL_PATH = "models/dataco_catboost_features_model.joblib"
RESULTS_PATH = "outputs/dataco_catboost_features_results.csv"
PREDICTIONS_PATH = "outputs/dataco_catboost_features_predictions.csv"

TARGET = "Late_delivery_risk"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("DATACO CATBOOST - FEATURE ENGINEERING EXPERIMENT")
print("=" * 70)

df = pd.read_csv(DATA_PATH, encoding="latin1")

print(f"\nOriginal dataset shape: {df.shape}")


# ============================================================
# DATE FEATURES
# ============================================================

date_col = "order date (DateOrders)"

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df["order_year"] = df[date_col].dt.year
df["order_month"] = df[date_col].dt.month
df["order_day"] = df[date_col].dt.day
df["order_dayofweek"] = df[date_col].dt.dayofweek
df["order_week"] = df[date_col].dt.isocalendar().week.astype(int)


# ============================================================
# FEATURE ENGINEERING
# ============================================================
#
# These features use ONLY information available before/during
# shipment planning.
#
# No:
#   Delivery Status
#   actual shipping duration
#   shipping date
#   Order Status
#
# This keeps the experiment leakage-safe.
# ============================================================

print("\nCreating interaction features...")

# Shipping mode + scheduled delivery duration
df["ShippingMode_ScheduledDays"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Days for shipment (scheduled)"].astype(str)
)

# Shipping mode + market
df["ShippingMode_Market"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Market"].astype(str)
)

# Shipping mode + order region
df["ShippingMode_Region"] = (
    df["Shipping Mode"].astype(str)
    + "_"
    + df["Order Region"].astype(str)
)

# Shipping mode + order month
df["ShippingMode_Month"] = (
    df["Shipping Mode"].astype(str)
    + "_M"
    + df["order_month"].astype(str)
)

# Market + scheduled days
df["Market_ScheduledDays"] = (
    df["Market"].astype(str)
    + "_"
    + df["Days for shipment (scheduled)"].astype(str)
)

# Market + order region
df["Market_Region"] = (
    df["Market"].astype(str)
    + "_"
    + df["Order Region"].astype(str)
)

# Region + scheduled days
df["Region_ScheduledDays"] = (
    df["Order Region"].astype(str)
    + "_"
    + df["Days for shipment (scheduled)"].astype(str)
)

print("Interaction features created:")
print("  ShippingMode_ScheduledDays")
print("  ShippingMode_Market")
print("  ShippingMode_Region")
print("  ShippingMode_Month")
print("  Market_ScheduledDays")
print("  Market_Region")
print("  Region_ScheduledDays")


# ============================================================
# REMOVE TARGET LEAKAGE / POST-OUTCOME FEATURES
# ============================================================

drop_columns = [

    # Target
    TARGET,

    # Direct leakage / post-outcome
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",

    # Customer personal information
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",

    # Customer/order identifiers
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",

    # Product/card identifiers
    "Product Card Id",
    "Order Item Cardprod Id",
    "Category Id",
    "Product Category Id",
    "Department Id",

    # High-cardinality / unnecessary identifiers
    "Order Zipcode",
    "Product Description",
    "Product Image",
    "Product Status",

    # Original date string
    "order date (DateOrders)"
]


# Only drop columns that actually exist
drop_columns = [
    col for col in drop_columns
    if col in df.columns
]

X = df.drop(columns=drop_columns)
y = df[TARGET]


# ============================================================
# IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_features = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nFinal feature information:")
print(f"Total features: {X.shape[1]}")
print(f"Numerical features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

for col in categorical_features:
    X[col] = X[col].fillna("Unknown").astype(str)

for col in numeric_features:
    X[col] = X[col].fillna(X[col].median())


# CatBoost requires categorical columns to be strings
for col in categorical_features:
    X[col] = X[col].astype(str)


# ============================================================
# 70 / 15 / 15 SPLIT
# ============================================================

print("\nCreating 70/15/15 split...")

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.30,
    stratify=y,
    random_state=RANDOM_STATE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"Training set:   {len(X_train):,}")
print(f"Validation set: {len(X_val):,}")
print(f"Test set:       {len(X_test):,}")


# ============================================================
# CATBOOST
# ============================================================

print("\nTraining CatBoost...")

model = CatBoostClassifier(

    iterations=5000,

    learning_rate=0.03,

    depth=9,

    l2_leaf_reg=7,

    loss_function="Logloss",

    eval_metric="AUC",

    random_seed=RANDOM_STATE,

    verbose=200,

    od_type="Iter",

    od_wait=200,

    allow_writing_files=False
)


model.fit(

    X_train,
    y_train,

    cat_features=categorical_features,

    eval_set=(X_val, y_val),

    use_best_model=True
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("FINAL FEATURE-ENGINEERED RESULTS")
print("=" * 70)

print(f"\nBest iteration: {model.get_best_iteration()}")

print(f"\nAccuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

print(f"\nModel saved to:")
print(MODEL_PATH)


# ============================================================
# SAVE RESULTS
# ============================================================

results = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],

    "Score": [
        accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["Actual"] = y_test.values
prediction_output["Predicted"] = y_pred
prediction_output["Risk_Probability"] = y_prob

prediction_output.to_csv(
    PREDICTIONS_PATH,
    index=False
)

print(f"\nResults saved to:")
print(RESULTS_PATH)

print(f"\nPredictions saved to:")
print(PREDICTIONS_PATH)


# ============================================================
# COMPARISON WITH CURRENT FINAL MODEL
# ============================================================

baseline_accuracy = 0.7958
baseline_auc = 0.8906

accuracy_change = accuracy - baseline_accuracy
auc_change = roc_auc - baseline_auc

print("\n" + "=" * 70)
print("COMPARISON WITH CURRENT 79.58% BASELINE")
print("=" * 70)

print(f"\nBaseline Accuracy : {baseline_accuracy * 100:.2f}%")
print(f"New Accuracy      : {accuracy * 100:.2f}%")

print(
    f"Accuracy Change   : "
    f"{accuracy_change * 100:+.2f} percentage points"
)

print(f"\nBaseline ROC-AUC  : {baseline_auc:.4f}")
print(f"New ROC-AUC       : {roc_auc:.4f}")

print(
    f"ROC-AUC Change    : "
    f"{auc_change:+.4f}"
)

if accuracy > baseline_accuracy:
    print("\nFeature engineering IMPROVED accuracy.")
elif accuracy < baseline_accuracy:
    print("\nFeature engineering REDUCED accuracy.")
else:
    print("\nFeature engineering produced the SAME accuracy.")

print("\nExperiment complete.")