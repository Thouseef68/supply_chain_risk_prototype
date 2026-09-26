import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

from xgboost import XGBClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"

MODEL_PATH = "models/dataco_xgboost_model.joblib"

TARGET = "Late_delivery_risk"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("DATACO SUPPLY CHAIN - XGBOOST TRAINING")
print("=" * 80)

print("\n[1] Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print(
    f"Rows    : {len(df):,}"
)

print(
    f"Columns : {len(df.columns)}"
)


# ============================================================
# DATE FEATURES
# ============================================================

print("\n[2] Creating date features...")

date_col = "order date (DateOrders)"

df["order_date"] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_day"] = df["order_date"].dt.day
df["order_dayofweek"] = df["order_date"].dt.dayofweek
df["order_week"] = (
    df["order_date"]
    .dt.isocalendar()
    .week
    .astype(float)
)

df.drop(
    columns=[
        date_col,
        "order_date"
    ],
    inplace=True
)


# ============================================================
# LEAKAGE / POST-OUTCOME COLUMNS
# ============================================================

LEAKAGE_COLUMNS = [

    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status"
]


# ============================================================
# PERSONAL / IDENTIFIER COLUMNS
# ============================================================

UNUSED_COLUMNS = [

    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",

    "Customer Id",
    "Order Customer Id",

    "Order Id",
    "Order Item Id",

    "Product Card Id",
    "Order Item Cardprod Id",

    "Category Id",
    "Product Category Id",

    "Department Id",

    "Order Zipcode",

    "Product Description",
    "Product Image",
    "Product Status"
]


DROP_COLUMNS = (
    LEAKAGE_COLUMNS +
    UNUSED_COLUMNS
)


DROP_COLUMNS = [
    col for col in DROP_COLUMNS
    if col in df.columns
]

df.drop(
    columns=DROP_COLUMNS,
    inplace=True
)

print(
    f"Removed {len(DROP_COLUMNS)} columns."
)


# ============================================================
# TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)


# ============================================================
# REMOVE VERY HIGH-CARDINALITY FIELDS
# ============================================================

print("\n[3] Preparing features...")

HIGH_CARDINALITY_COLUMNS = [
    "Customer City",
    "Customer State",
    "Order City",
    "Order State",
    "Product Name"
]

for col in HIGH_CARDINALITY_COLUMNS:

    if col in X.columns:

        X.drop(
            columns=[col],
            inplace=True
        )

print(
    "High-cardinality fields removed."
)


# ============================================================
# FEATURE TYPES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

print(
    "\nNumerical features:",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training: {len(X_train):,}"
)

print(
    f"Testing : {len(X_test):,}"
)


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                min_frequency=5
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# XGBOOST
# ============================================================

print("\n[5] Creating XGBoost model...")

model = XGBClassifier(

    n_estimators=500,

    learning_rate=0.05,

    max_depth=7,

    min_child_weight=3,

    subsample=0.85,

    colsample_bytree=0.85,

    objective="binary:logistic",

    eval_metric="logloss",

    tree_method="hist",

    n_jobs=-1,

    random_state=42
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)


# ============================================================
# TRAIN
# ============================================================

print("\n[6] Training XGBoost...")
print("This may take some time with 180k records.")

pipeline.fit(
    X_train,
    y_train
)


# ============================================================
# PREDICTIONS
# ============================================================

print("\n[7] Evaluating model...")

predictions = pipeline.predict(
    X_test
)

probabilities = pipeline.predict_proba(
    X_test
)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

precision = precision_score(
    y_test,
    predictions
)

recall = recall_score(
    y_test,
    predictions
)

f1 = f1_score(
    y_test,
    predictions
)

roc_auc = roc_auc_score(
    y_test,
    probabilities
)


print("\n" + "=" * 80)
print("XGBOOST RESULTS")
print("=" * 80)

print(
    f"Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Precision: {precision * 100:.2f}%"
)

print(
    f"Recall   : {recall * 100:.2f}%"
)

print(
    f"F1 Score : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print("\nCONFUSION MATRIX")
print("-" * 50)

cm = confusion_matrix(
    y_test,
    predictions
)

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nCLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        y_test,
        predictions,
        digits=4
    )
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = pd.DataFrame([
    {
        "Model": "XGBoost",
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": roc_auc
    }
])

metrics.to_csv(
    "outputs/dataco_xgboost_results.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\nModel saved:")
print(MODEL_PATH)


# ============================================================
# SAVE TEST PREDICTIONS
# ============================================================

test_results = X_test.copy()

test_results["Actual_Risk"] = y_test.values

test_results["Predicted_Risk"] = predictions

test_results["Risk_Probability"] = probabilities

test_results.to_csv(
    "outputs/dataco_xgboost_predictions.csv",
    index=False
)

print(
    "\nTest predictions saved:"
)

print(
    "outputs/dataco_xgboost_predictions.csv"
)


print("\n" + "=" * 80)
print("XGBOOST TRAINING COMPLETE")
print("=" * 80)