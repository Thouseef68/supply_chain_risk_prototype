import pandas as pd
import numpy as np
import joblib

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

from catboost import CatBoostClassifier


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/dataco_catboost_model.joblib"
RESULTS_PATH = "outputs/dataco_catboost_results.csv"
PREDICTIONS_PATH = "outputs/dataco_catboost_predictions.csv"

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("DATACO SUPPLY CHAIN - CATBOOST")
print("=" * 80)


# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1] Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

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
df["order_week"] = (
    df[date_col].dt.isocalendar().week.astype("float")
)


# ============================================================
# 3. TARGET
# ============================================================

TARGET = "Late_delivery_risk"

y = df[TARGET].astype(int)


# ============================================================
# 4. REMOVE LEAKAGE / IDENTIFIERS
# ============================================================

remove_columns = [

    # --------------------------------------------------------
    # Post-outcome / leakage
    # --------------------------------------------------------

    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",

    # --------------------------------------------------------
    # Personal information
    # --------------------------------------------------------

    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",

    # --------------------------------------------------------
    # Identifiers
    # --------------------------------------------------------

    "Customer Id",
    "Order Customer Id",
    "Order Id",
    "Order Item Id",
    "Product Card Id",
    "Order Item Cardprod Id",

    # --------------------------------------------------------
    # Category IDs
    # --------------------------------------------------------

    "Category Id",
    "Product Category Id",
    "Department Id",

    # --------------------------------------------------------
    # Missing / useless
    # --------------------------------------------------------

    "Order Zipcode",
    "Product Description",
    "Product Image",
    "Product Status",

    # --------------------------------------------------------
    # Original raw date
    # --------------------------------------------------------

    "order date (DateOrders)",

    # --------------------------------------------------------
    # High-cardinality fields
    # --------------------------------------------------------

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

df = df.drop(
    columns=remove_columns
)

print(f"Removed {len(remove_columns)} columns.")


# ============================================================
# 5. PREPARE FEATURES
# ============================================================

print("\n[3] Preparing features...")

X = df.drop(
    columns=[TARGET]
).copy()


# ============================================================
# 6. IDENTIFY CATEGORICAL FEATURES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["int64", "float64", "int32", "float32"]
).columns.tolist()

print(f"Numerical features   : {len(numeric_features)}")
print(f"Categorical features : {len(categorical_features)}")


# ============================================================
# 7. CATBOOST REQUIRES CATEGORICAL VALUES TO BE NON-NULL
# ============================================================

for col in categorical_features:

    X[col] = X[col].astype(str)

    X[col] = X[col].replace(
        ["nan", "None", ""],
        "Unknown"
    )


# ============================================================
# 8. NUMERICAL MISSING VALUES
# ============================================================

for col in numeric_features:

    X[col] = X[col].replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# 9. TRAIN / TEST SPLIT
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
# 10. CATEGORICAL COLUMN INDICES
# ============================================================

cat_indices = [
    X_train.columns.get_loc(col)
    for col in categorical_features
]


# ============================================================
# 11. CREATE CATBOOST MODEL
# ============================================================

print("\n[5] Creating CatBoost model...")

model = CatBoostClassifier(

    iterations=4000,

    learning_rate=0.03,

    depth=9,

    l2_leaf_reg=7,

    loss_function="Logloss",

    eval_metric="AUC",

    random_seed=RANDOM_STATE,

    verbose=100,

    thread_count=-1,

    allow_writing_files=False
)


# ============================================================
# 12. TRAIN
# ============================================================

print("\n[6] Training CatBoost...")
print("This may take some time with 180k records.")

model.fit(

    X_train,

    y_train,

    cat_features=cat_indices,

    eval_set=(X_test, y_test),

    use_best_model=True
)


# ============================================================
# 13. PREDICTION
# ============================================================

print("\n[7] Evaluating model...")

y_probability = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_probability >= 0.50
).astype(int)


# ============================================================
# 14. METRICS
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

cm = confusion_matrix(
    y_test,
    y_pred
)


# ============================================================
# 15. RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("CATBOOST RESULTS")
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
# 16. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{

    "model": "CatBoost",

    "accuracy": accuracy,

    "precision": precision,

    "recall": recall,

    "f1_score": f1,

    "roc_auc": roc_auc

}])


results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# 17. SAVE PREDICTIONS
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
# 18. SAVE MODEL
# ============================================================

joblib.dump(
    {
        "model": model,
        "categorical_features": categorical_features,
        "numeric_features": numeric_features,
        "threshold": 0.50
    },
    MODEL_PATH
)


# ============================================================
# COMPLETE
# ============================================================

print("\n")
print("=" * 80)
print("CATBOOST TRAINING COMPLETE")
print("=" * 80)

print("\nModel saved:")
print(MODEL_PATH)

print("\nResults saved:")
print(RESULTS_PATH)

print("\nPredictions saved:")
print(PREDICTIONS_PATH)