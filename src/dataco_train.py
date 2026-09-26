import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)


# ============================================================
# CONFIG
# ============================================================

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"
MODEL_DIR = "models"

TARGET = "Late_delivery_risk"

os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# LOAD
# ============================================================

print("=" * 80)
print("DATACO SUPPLY CHAIN - MODEL TRAINING")
print("=" * 80)

print("\n[1] Loading dataset...")

df = pd.read_csv(
    DATA_PATH,
    encoding="latin1"
)

print(
    f"Loaded {len(df):,} rows and {len(df.columns)} columns."
)


# ============================================================
# DATE FEATURES
# ============================================================

print("\n[2] Creating date features...")

df["order_date"] = pd.to_datetime(
    df["order date (DateOrders)"],
    errors="coerce"
)

df["order_year"] = df["order_date"].dt.year
df["order_month"] = df["order_date"].dt.month
df["order_dayofweek"] = df["order_date"].dt.dayofweek
df["order_week"] = (
    df["order_date"]
    .dt.isocalendar()
    .week
    .astype(float)
)

df.drop(
    columns=[
        "order_date",
        "order date (DateOrders)"
    ],
    inplace=True
)


# ============================================================
# LEAKAGE + USELESS COLUMNS
# ============================================================

DROP_COLUMNS = [

    # --------------------------------------------------------
    # POST-OUTCOME / LEAKAGE
    # --------------------------------------------------------

    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",

    # --------------------------------------------------------
    # PERSONAL INFORMATION
    # --------------------------------------------------------

    "Customer Email",
    "Customer Fname",
    "Customer Lname",
    "Customer Password",
    "Customer Street",
    "Customer Zipcode",

    # --------------------------------------------------------
    # IDENTIFIERS
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # UNHELPFUL PRODUCT METADATA
    # --------------------------------------------------------

    "Product Description",
    "Product Image",
    "Product Status"
]


existing_drop = [
    col for col in DROP_COLUMNS
    if col in df.columns
]

df.drop(
    columns=existing_drop,
    inplace=True
)

print(
    f"Removed {len(existing_drop)} columns."
)


# ============================================================
# TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
)

y = df[TARGET].astype(int)


# ============================================================
# REMOVE EXTREME HIGH-CARDINALITY FEATURES
# ============================================================

print("\n[3] Feature selection...")

# These fields have many unique values and can create
# unnecessary dimensionality without strong evidence
# that they improve the pre-shipment prediction.

high_cardinality_remove = [
    "Customer City",
    "Customer State",
    "Order City",
    "Order State",
    "Product Name"
]

for col in high_cardinality_remove:

    if col in X.columns:
        X.drop(
            columns=[col],
            inplace=True
        )

print(
    "High-cardinality location/product fields removed."
)


# ============================================================
# IDENTIFY FEATURE TYPES
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

print("\n[4] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training rows: {len(X_train):,}"
)

print(
    f"Testing rows : {len(X_test):,}"
)


# ============================================================
# PREPROCESSOR
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)


categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
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
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        C=1.0,
        class_weight="balanced",
        n_jobs=-1
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=250,
        max_depth=18,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    ),

    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.08,
        max_leaf_nodes=31,
        random_state=42
    )
}


# ============================================================
# TRAIN
# ============================================================

results = []

best_model = None
best_score = -1
best_name = None


for name, model in models.items():

    print("\n" + "=" * 80)
    print(f"TRAINING: {name}")
    print("=" * 80)

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

    pipeline.fit(
        X_train,
        y_train
    )

    predictions = pipeline.predict(
        X_test
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

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

    auc = roc_auc_score(
        y_test,
        probabilities
    )

    print(
        f"\nAccuracy : {accuracy * 100:.2f}%"
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
        f"ROC-AUC  : {auc:.4f}"
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "ROC_AUC": auc
    })

    # Use ROC-AUC as the primary model-selection metric.
    if auc > best_score:

        best_score = auc
        best_model = pipeline
        best_name = name


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
).sort_values(
    "ROC_AUC",
    ascending=False
)

print("\n" + "=" * 80)
print("MODEL COMPARISON")
print("=" * 80)

print(
    results_df.to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results_df.to_csv(
    "outputs/dataco_model_comparison.csv",
    index=False
)


# ============================================================
# SAVE BEST MODEL
# ============================================================

model_path = (
    f"{MODEL_DIR}/dataco_late_delivery_model.joblib"
)

joblib.dump(
    best_model,
    model_path
)

print("\n" + "=" * 80)
print("BEST MODEL")
print("=" * 80)

print(
    "Model:",
    best_name
)

print(
    f"ROC-AUC: {best_score:.4f}"
)

print(
    f"Saved: {model_path}"
)


# ============================================================
# FINAL CLASSIFICATION REPORT
# ============================================================

final_predictions = best_model.predict(
    X_test
)

print("\nCLASSIFICATION REPORT")
print("=" * 80)

print(
    classification_report(
        y_test,
        final_predictions,
        digits=4
    )
)


# ============================================================
# SAVE TEST DATA
# ============================================================

test_output = X_test.copy()

test_output["Actual_Risk"] = y_test.values
test_output["Predicted_Risk"] = final_predictions

test_output.to_csv(
    "outputs/dataco_test_predictions.csv",
    index=False
)

print("\nTest predictions saved.")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)