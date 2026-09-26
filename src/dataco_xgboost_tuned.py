import pandas as pd
import numpy as np
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
MODEL_PATH = "models/dataco_xgboost_tuned_model.joblib"
RESULTS_PATH = "outputs/dataco_xgboost_tuned_results.csv"
PREDICTIONS_PATH = "outputs/dataco_xgboost_tuned_predictions.csv"

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("DATACO SUPPLY CHAIN - TUNED XGBOOST")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(DATA_PATH, encoding="latin1")

print(f"Rows    : {df.shape[0]:,}")
print(f"Columns : {df.shape[1]}")


# ============================================================
# 2. DATE FEATURES
# ============================================================

print("\n[2] Creating date features...")

date_col = "order date (DateOrders)"

df[date_col] = pd.to_datetime(
    df[date_col],
    errors="coerce"
)

df["order_year"] = df[date_col].dt.year
df["order_month"] = df[date_col].dt.month
df["order_day"] = df[date_col].dt.day
df["order_dayofweek"] = df[date_col].dt.dayofweek
df["order_week"] = df[date_col].dt.isocalendar().week.astype("float")


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "Late_delivery_risk"

y = df[TARGET].astype(int)


# ============================================================
# 4. REMOVE LEAKAGE / POST-OUTCOME FEATURES
# ============================================================

leakage_columns = [
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status"
]


# ============================================================
# 5. REMOVE PERSONAL / IDENTIFIER / USELESS FEATURES
# ============================================================

remove_columns = [

    # Leakage / post-outcome
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",

    # Personal information
    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",

    # Identifiers
    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Product Card Id",
    "Order Item Cardprod Id",

    # Category IDs
    "Category Id",
    "Product Category Id",
    "Department Id",

    # Missing / useless
    "Order Zipcode",
    "Product Description",
    "Product Image",
    "Product Status",

    # Original raw date
    "order date (DateOrders)",

    # High-cardinality fields
    "Customer City",
    "Customer State",
    "Order City",
    "Order State",
    "Product Name"
]

remove_columns = [
    col for col in remove_columns
    if col in df.columns
]

df = df.drop(columns=remove_columns)

print(f"Removed {len(remove_columns)} columns.")


# ============================================================
# 6. PREPARE FEATURES
# ============================================================

print("\n[3] Preparing features...")

X = df.drop(columns=[TARGET])

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

print(f"Numerical features   : {len(numeric_features)}")
print(f"Categorical features : {len(categorical_features)}")


# ============================================================
# 7. PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(
            handle_unknown="ignore",
            min_frequency=3
        )
    )
])


preprocessor = ColumnTransformer([
    (
        "num",
        numeric_pipeline,
        numeric_features
    ),
    (
        "cat",
        categorical_pipeline,
        categorical_features
    )
])


# ============================================================
# 8. TRAIN / TEST SPLIT
# ============================================================

print("\n[4] Splitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print(f"Training: {len(X_train):,}")
print(f"Testing : {len(X_test):,}")


# ============================================================
# 9. XGBOOST
# ============================================================

print("\n[5] Creating tuned XGBoost model...")

model = XGBClassifier(

    # More boosting rounds + lower learning rate
    n_estimators=900,
    learning_rate=0.035,

    # Tree complexity
    max_depth=8,
    min_child_weight=2,

    # Randomization
    subsample=0.90,
    colsample_bytree=0.90,

    # Regularization
    gamma=0.05,
    reg_alpha=0.05,
    reg_lambda=1.5,

    # Classification
    objective="binary:logistic",
    eval_metric="logloss",

    # Performance
    tree_method="hist",
    n_jobs=-1,

    random_state=RANDOM_STATE
)


# ============================================================
# 10. TRANSFORM DATA
# ============================================================

print("\n[6] Transforming features...")

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(
    "Processed training shape:",
    X_train_processed.shape
)

print(
    "Processed testing shape :",
    X_test_processed.shape
)


# ============================================================
# 11. TRAIN
# ============================================================

print("\n[7] Training tuned XGBoost...")
print("This may take some time...")

model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# 12. PREDICTION
# ============================================================

print("\n[8] Evaluating model...")

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]

# Standard classification threshold
threshold = 0.50

y_pred = (
    y_probability >= threshold
).astype(int)


# ============================================================
# 13. METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probability)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 14. RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("TUNED XGBOOST RESULTS")
print("=" * 80)

print(f"Accuracy : {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall   : {recall * 100:.2f}%")
print(f"F1 Score : {f1 * 100:.2f}%")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nCONFUSION MATRIX")
print("-" * 50)
print(cm)

print("\nCLASSIFICATION REPORT")
print("-" * 50)

print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================================
# 15. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{
    "model": "Tuned XGBoost",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "roc_auc": roc_auc,
    "threshold": threshold
}])

results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# 16. SAVE PREDICTIONS
# ============================================================

prediction_output = X_test.copy()

prediction_output["actual_risk"] = y_test.values
prediction_output["risk_probability"] = y_probability
prediction_output["predicted_risk"] = y_pred

prediction_output.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# 17. SAVE MODEL + PREPROCESSOR
# ============================================================

model_bundle = {
    "preprocessor": preprocessor,
    "model": model,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "threshold": threshold
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("TUNED XGBOOST TRAINING COMPLETE")
print("=" * 80)

print(f"\nModel saved:")
print(MODEL_PATH)

print(f"\nResults saved:")
print(RESULTS_PATH)

print(f"\nPredictions saved:")
print(PREDICTIONS_PATH)