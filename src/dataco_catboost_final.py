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

MODEL_PATH = "models/dataco_catboost_final_model.joblib"
RESULTS_PATH = "outputs/dataco_catboost_final_results.csv"
PREDICTIONS_PATH = "outputs/dataco_catboost_final_predictions.csv"

RANDOM_STATE = 42


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("DATACO SUPPLY CHAIN - FINAL CATBOOST EVALUATION")
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
# 6. IDENTIFY FEATURES
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
# 7. CLEAN CATEGORICAL VALUES
# ============================================================

for col in categorical_features:

    X[col] = X[col].astype(str)

    X[col] = X[col].replace(
        ["nan", "None", ""],
        "Unknown"
    )


# ============================================================
# 8. CLEAN NUMERICAL VALUES
# ============================================================

for col in numeric_features:

    X[col] = X[col].replace(
        [np.inf, -np.inf],
        np.nan
    )


# ============================================================
# 9. FIRST SPLIT
#    70% TRAIN
#    30% TEMP
# ============================================================

print("\n[4] Creating train / validation / test split...")

X_train, X_temp, y_train, y_temp = train_test_split(

    X,
    y,

    test_size=0.30,

    random_state=RANDOM_STATE,

    stratify=y
)


# ============================================================
# 10. SECOND SPLIT
#     15% VALIDATION
#     15% TEST
# ============================================================

X_val, X_test, y_val, y_test = train_test_split(

    X_temp,
    y_temp,

    test_size=0.50,

    random_state=RANDOM_STATE,

    stratify=y_temp
)


print(f"Training   : {len(X_train):,}")
print(f"Validation : {len(X_val):,}")
print(f"Test       : {len(X_test):,}")


# ============================================================
# 11. CATEGORICAL COLUMN INDICES
# ============================================================

cat_indices = [
    X_train.columns.get_loc(col)
    for col in categorical_features
]


# ============================================================
# 12. CREATE CATBOOST
# ============================================================

print("\n[5] Creating final CatBoost model...")

model = CatBoostClassifier(

    iterations=5000,

    learning_rate=0.03,

    depth=9,

    l2_leaf_reg=7,

    loss_function="Logloss",

    eval_metric="AUC",

    random_seed=RANDOM_STATE,

    verbose=100,

    thread_count=-1,

    allow_writing_files=False,

    # Stop if validation AUC stops improving
    od_type="Iter",

    od_wait=200
)


# ============================================================
# 13. TRAIN USING VALIDATION SET
# ============================================================

print("\n[6] Training CatBoost...")
print("Validation set is used for model selection.")
print("Final test set remains untouched.")

model.fit(

    X_train,

    y_train,

    cat_features=cat_indices,

    eval_set=(X_val, y_val),

    use_best_model=True
)


# ============================================================
# 14. VALIDATION INFORMATION
# ============================================================

print("\n")
print("=" * 80)
print("VALIDATION / TRAINING INFORMATION")
print("=" * 80)

print(
    f"Best iteration: {model.get_best_iteration()}"
)

print(
    f"Best validation AUC: "
    f"{model.get_best_score()['validation']['AUC']:.4f}"
)


# ============================================================
# 15. FINAL TEST PREDICTION
# ============================================================

print("\n[7] Evaluating on untouched test set...")

y_probability = model.predict_proba(
    X_test
)[:, 1]

y_pred = (
    y_probability >= 0.50
).astype(int)


# ============================================================
# 16. FINAL TEST METRICS
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
# 17. FINAL RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("FINAL UNTOUCHED TEST RESULTS")
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
# 18. SAVE RESULTS
# ============================================================

results = pd.DataFrame([{

    "model": "CatBoost Final",

    "train_rows": len(X_train),

    "validation_rows": len(X_val),

    "test_rows": len(X_test),

    "best_iteration": model.get_best_iteration(),

    "validation_auc": model.get_best_score()["validation"]["AUC"],

    "test_accuracy": accuracy,

    "test_precision": precision,

    "test_recall": recall,

    "test_f1": f1,

    "test_roc_auc": roc_auc

}])


results.to_csv(
    RESULTS_PATH,
    index=False
)


# ============================================================
# 19. SAVE TEST PREDICTIONS
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
# 20. SAVE FINAL MODEL
# ============================================================

model_bundle = {

    "model": model,

    "categorical_features": categorical_features,

    "numeric_features": numeric_features,

    "threshold": 0.50,

    "best_iteration": model.get_best_iteration(),

    "validation_auc": model.get_best_score()["validation"]["AUC"]

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
print("FINAL CATBOOST EVALUATION COMPLETE")
print("=" * 80)

print("\nFinal model saved:")
print(MODEL_PATH)

print("\nFinal results saved:")
print(RESULTS_PATH)

print("\nTest predictions saved:")
print(PREDICTIONS_PATH)