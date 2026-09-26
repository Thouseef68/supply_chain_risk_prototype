import pandas as pd
import numpy as np
import glob
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# LOAD DATA
# ============================================================

file = glob.glob("data/*.csv")[0]

df = pd.read_csv(file)

print("=" * 70)
print("SUPPLY CHAIN MODEL OPTIMIZATION")
print("=" * 70)

print("\nDataset:", file)
print("Shape:", df.shape)


# ============================================================
# DATE FEATURES
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day_of_Week"] = df["Date"].dt.dayofweek

df = df.drop(
    columns=["Shipment_ID", "Date"]
)


# ============================================================
# X / Y
# ============================================================

X = df.drop(
    columns=["Disruption_Occurred"]
)

y = df["Disruption_Occurred"]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "num",
            StandardScaler(),
            numerical
        ),

        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical
        )

    ]

)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        C=0.1,
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ),

    "Logistic Regression C1": LogisticRegression(
        C=1.0,
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Extra Trees": ExtraTreesClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        max_features=1.0,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.9,
        random_state=42
    )
}


# ============================================================
# TRAIN
# ============================================================

results = []

trained = {}


for name, model in models.items():

    print("\n" + "-" * 70)
    print(name)
    print("-" * 70)

    pipeline = Pipeline(

        steps=[
            ("preprocessing", preprocessor),
            ("model", model)
        ]

    )

    pipeline.fit(
        X_train,
        y_train
    )

    probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    # Standard threshold
    predictions = (
        probabilities >= 0.50
    ).astype(int)

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
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"ROC-AUC  : {auc:.4f}"
    )

    results.append({

        "Model": name,

        "Threshold": 0.50,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1": f1,

        "ROC_AUC": auc

    })

    trained[name] = (
        pipeline,
        probabilities
    )


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

print("\n")
print("=" * 70)
print("THRESHOLD OPTIMIZATION")
print("=" * 70)


for name, (
    pipeline,
    probabilities
) in trained.items():

    best_accuracy = 0
    best_threshold = 0.50

    for threshold in np.arange(
        0.20,
        0.81,
        0.01
    ):

        predictions = (
            probabilities >= threshold
        ).astype(int)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_threshold = threshold

    predictions = (
        probabilities >= best_threshold
    ).astype(int)

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
        f"\n{name}"
    )

    print(
        f"Best threshold : {best_threshold:.2f}"
    )

    print(
        f"Accuracy       : {best_accuracy:.4f}"
    )

    print(
        f"Precision      : {precision:.4f}"
    )

    print(
        f"Recall         : {recall:.4f}"
    )

    print(
        f"F1             : {f1:.4f}"
    )

    print(
        f"ROC-AUC        : {auc:.4f}"
    )


# ============================================================
# FINAL COMPARISON
# ============================================================

print("\n")
print("=" * 70)
print("IMPORTANT")
print("=" * 70)

print(
    """
Do NOT choose a model simply because it has the highest
accuracy.

For a supply-chain disruption system, missing an actual
disruption can be costly.

Therefore we will consider:

1. Accuracy
2. Recall
3. Precision
4. F1 Score
5. ROC-AUC
6. Business usefulness
"""
)

print("\nOptimization completed.")