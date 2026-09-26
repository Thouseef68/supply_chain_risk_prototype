import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"
MODEL_PATH = "models/dataco_decision_tree_model.joblib"

RANDOM_STATE = 42

print("=" * 80)
print("DATACO SUPPLY CHAIN - DECISION TREE")
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
# 4. REMOVE LEAKAGE / IDENTIFIERS
# ============================================================

remove_columns = [

    # Post-outcome / leakage
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

    # Original date
    "order date (DateOrders)",

    # High-cardinality
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
# 5. FEATURES
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
# 6. PREPROCESSING
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
# 7. SPLIT
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
# 8. TRANSFORM
# ============================================================

print("\n[5] Transforming features...")

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
# 9. DECISION TREE
# ============================================================

print("\n[6] Creating Decision Tree...")

model = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=15,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features=None,
    class_weight=None,
    random_state=RANDOM_STATE
)


# ============================================================
# 10. TRAIN
# ============================================================

print("\n[7] Training Decision Tree...")

model.fit(
    X_train_processed,
    y_train
)


# ============================================================
# 11. PREDICT
# ============================================================

print("\n[8] Evaluating model...")

y_pred = model.predict(
    X_test_processed
)

y_probability = model.predict_proba(
    X_test_processed
)[:, 1]


# ============================================================
# 12. METRICS
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
# 13. RESULTS
# ============================================================

print("\n")
print("=" * 80)
print("DECISION TREE RESULTS")
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
# 14. SAVE MODEL
# ============================================================

model_bundle = {
    "preprocessor": preprocessor,
    "model": model,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features
}

joblib.dump(
    model_bundle,
    MODEL_PATH
)

print("\nModel saved:")
print(MODEL_PATH)

print("\n")
print("=" * 80)
print("DECISION TREE TRAINING COMPLETE")
print("=" * 80)