import os
import hashlib
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"

MODEL_PATH = "models/dataco_catboost_features_model.joblib"

PREDICTIONS_PATH = (
    "outputs/dataco_catboost_features_predictions.csv"
)

RANDOM_STATE = 42

TARGET = "Late_delivery_risk"


# ============================================================
# HELPERS
# ============================================================

def file_hash(path):

    sha256 = hashlib.sha256()

    with open(path, "rb") as f:

        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):

            sha256.update(block)

    return sha256.hexdigest()


def status(ok):

    return "PASS" if ok else "ATTENTION"


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("FINAL MODEL INTEGRITY / LEAKAGE / OVERFITTING AUDIT")
print("=" * 80)

print("\nThis audit does NOT retrain the model.")


# ============================================================
# 1. FILE EXISTENCE
# ============================================================

print("\n" + "=" * 80)
print("1. REQUIRED FILE CHECK")
print("=" * 80)

files = [
    DATA_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH
]

all_exist = True

for path in files:

    exists = os.path.exists(path)

    print(
        f"{status(exists):<10} {path}"
    )

    if not exists:
        all_exist = False


# ============================================================
# 2. LOAD MODEL
# ============================================================

print("\n" + "=" * 80)
print("2. MODEL CHECK")
print("=" * 80)

model = joblib.load(MODEL_PATH)

print("PASS       Model loaded successfully")

model_features = list(
    model.feature_names_
)

print(
    f"PASS       Model feature count: "
    f"{len(model_features)}"
)


# ============================================================
# 3. LOAD ORIGINAL DATA
# ============================================================

print("\n" + "=" * 80)
print("3. ORIGINAL DATASET CHECK")
print("=" * 80)

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print(
    f"PASS       Dataset rows: "
    f"{len(df):,}"
)

print(
    f"PASS       Dataset columns: "
    f"{len(df.columns)}"
)

print(
    f"PASS       Target column exists: "
    f"{TARGET in df.columns}"
)


# ============================================================
# 4. TARGET DISTRIBUTION
# ============================================================

print("\n" + "=" * 80)
print("4. TARGET DISTRIBUTION")
print("=" * 80)

target_counts = df[TARGET].value_counts()

print(target_counts)

print(
    "\nTarget percentages:"
)

print(
    (df[TARGET].value_counts(normalize=True) * 100)
    .round(2)
)


# ============================================================
# 5. DUPLICATE ROW CHECK
# ============================================================

print("\n" + "=" * 80)
print("5. DUPLICATE CHECK")
print("=" * 80)

duplicate_count = df.duplicated().sum()

print(
    f"{status(duplicate_count == 0):<10}"
    f"Duplicate complete rows: {duplicate_count:,}"
)


# ============================================================
# 6. DIRECT LEAKAGE CHECK
# ============================================================

print("\n" + "=" * 80)
print("6. DIRECT TARGET LEAKAGE CHECK")
print("=" * 80)

leakage_columns = [

    "Late_delivery_risk",
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status"
]

direct_leakage_found = []

for col in leakage_columns:

    if col in model_features:

        direct_leakage_found.append(col)

        print(
            f"FAIL       {col} is in model features"
        )

    else:

        print(
            f"PASS       {col} is NOT in model features"
        )


# ============================================================
# 7. SUSPICIOUS TARGET-NAMED FEATURES
# ============================================================

print("\n" + "=" * 80)
print("7. TARGET-NAME FEATURE CHECK")
print("=" * 80)

suspicious_terms = [

    "late",
    "risk",
    "delivery",
    "delay",
    "actual_shipping",
    "real_shipping"

]

suspicious_features = []

for feature in model_features:

    lower = feature.lower()

    if any(
        term in lower
        for term in suspicious_terms
    ):

        suspicious_features.append(feature)


if suspicious_features:

    print(
        "ATTENTION  Potentially suspicious names:"
    )

    for feature in suspicious_features:

        print(
            f"           {feature}"
        )

else:

    print(
        "PASS       No obviously target-derived "
        "feature names found"
    )


# ============================================================
# 8. SAVED TEST PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("8. SAVED TEST PREDICTION CHECK")
print("=" * 80)

pred = pd.read_csv(
    PREDICTIONS_PATH
)

required_prediction_columns = [

    "Actual",
    "Predicted",
    "Risk_Probability"

]

prediction_columns_ok = all(
    col in pred.columns
    for col in required_prediction_columns
)

print(
    f"{status(prediction_columns_ok):<10}"
    "Prediction output columns"
)

print(
    f"PASS       Test rows: {len(pred):,}"
)


# ============================================================
# 9. RECALCULATE METRICS
# ============================================================

print("\n" + "=" * 80)
print("9. METRIC REPRODUCTION")
print("=" * 80)

y_true = pred["Actual"]

y_pred = pred["Predicted"]

y_prob = pred["Risk_Probability"]

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

auc = roc_auc_score(
    y_true,
    y_prob
)

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
    f"F1       : {f1 * 100:.2f}%"
)

print(
    f"ROC-AUC  : {auc:.4f}"
)


# ============================================================
# 10. MODEL PREDICTION REPRODUCTION
# ============================================================

print("\n" + "=" * 80)
print("10. SAVED PREDICTION REPRODUCTION")
print("=" * 80)

# The prediction CSV contains all model input columns
# followed by Actual / Predicted / Risk_Probability.

available_model_columns = [
    col for col in model_features
    if col in pred.columns
]

same_feature_set = (
    len(available_model_columns)
    == len(model_features)
)

print(
    f"{status(same_feature_set):<10}"
    f"All {len(model_features)} model features "
    f"present in prediction file"
)


if same_feature_set:

    X_saved = pred[model_features]

    # Convert categorical/object columns to strings
    for col in model_features:

        if X_saved[col].dtype == "object":

            X_saved[col] = (
                X_saved[col]
                .fillna("Unknown")
                .astype(str)
            )

    try:

        regenerated_prob = (
            model.predict_proba(X_saved)[:, 1]
        )

        regenerated_pred = (
            regenerated_prob >= 0.5
        ).astype(int)

        probability_difference = np.max(
            np.abs(
                regenerated_prob
                - y_prob.to_numpy()
            )
        )

        prediction_difference = np.sum(
            regenerated_pred
            != y_pred.to_numpy()
        )

        print(
            f"Maximum probability difference: "
            f"{probability_difference:.12f}"
        )

        print(
            f"Different class predictions: "
            f"{prediction_difference:,}"
        )

        print(
            status(
                probability_difference < 1e-6
                and prediction_difference == 0
            ),
            "Saved predictions reproduce from model"
        )

    except Exception as e:

        print(
            "ATTENTION  Could not reproduce "
            f"predictions: {e}"
        )


# ============================================================
# 11. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("11. CONFUSION MATRIX")
print("=" * 80)

cm = confusion_matrix(
    y_true,
    y_pred
)

print(
    """
                  Predicted
                  0       1
Actual 0
Actual 1
"""
)

print(cm)


# ============================================================
# 12. PREDICTION PROBABILITY SANITY
# ============================================================

print("\n" + "=" * 80)
print("12. PROBABILITY SANITY CHECK")
print("=" * 80)

probability_valid = (
    y_prob.min() >= 0
    and y_prob.max() <= 1
)

print(
    f"{status(probability_valid):<10}"
    f"Probability range: "
    f"{y_prob.min():.6f} - {y_prob.max():.6f}"
)


# ============================================================
# 13. EXTREME PREDICTIONS
# ============================================================

print("\n" + "=" * 80)
print("13. CONFIDENCE DISTRIBUTION")
print("=" * 80)

print(
    f"Probability <= 0.05 : "
    f"{(y_prob <= 0.05).sum():,}"
)

print(
    f"Probability >= 0.95 : "
    f"{(y_prob >= 0.95).sum():,}"
)

print(
    f"Total test samples   : "
    f"{len(y_prob):,}"
)


# ============================================================
# 14. TRAIN / VALIDATION / TEST EXPECTED SIZES
# ============================================================

print("\n" + "=" * 80)
print("14. DATA SPLIT CHECK")
print("=" * 80)

expected_train = int(
    len(df) * 0.70
)

expected_test = int(
    len(df) * 0.15
)

expected_validation = (
    len(df)
    - expected_train
    - expected_test
)

print(
    f"Dataset total : {len(df):,}"
)

print(
    f"Expected train: {expected_train:,}"
)

print(
    f"Expected val  : {expected_validation:,}"
)

print(
    f"Expected test : {expected_test:,}"
)

print(
    f"Saved test    : {len(pred):,}"
)

print(
    status(
        len(pred) == expected_test
    ),
    "Test-set size"
)


# ============================================================
# 15. FILE HASHES
# ============================================================

print("\n" + "=" * 80)
print("15. FILE INTEGRITY HASHES")
print("=" * 80)

for path in [
    DATA_PATH,
    MODEL_PATH,
    PREDICTIONS_PATH
]:

    print(
        f"\n{path}"
    )

    print(
        file_hash(path)
    )


# ============================================================
# 16. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("FINAL AUDIT SUMMARY")
print("=" * 80)

print(
    f"""
Dataset:
    {len(df):,} rows
    {len(df.columns)} original columns

Model:
    {len(model_features)} input features

Evaluation:
    {len(pred):,} saved test samples

Performance:
    Accuracy  = {accuracy * 100:.2f}%
    Precision = {precision * 100:.2f}%
    Recall    = {recall * 100:.2f}%
    F1        = {f1 * 100:.2f}%
    ROC-AUC   = {auc:.4f}

Direct leakage columns found:
    {len(direct_leakage_found)}

Duplicate original rows:
    {duplicate_count:,}
"""
)

if (
    len(direct_leakage_found) == 0
    and duplicate_count == 0
    and
    prediction_columns_ok
):

    print(
        "PASS: No direct target leakage detected "
        "by this audit."
    )

else:

    print(
        "ATTENTION: One or more integrity checks "
        "need investigation."
    )


print("\n" + "=" * 80)
print("AUDIT FINISHED")
print("=" * 80)

print(
    """
IMPORTANT:
This audit can detect direct leakage and verify
the reported test result, but it cannot mathematically
prove zero statistical overfitting.

A stronger generalization test would require
evaluation on an independent dataset or a new
retrained holdout/cross-validation experiment.
"""
)