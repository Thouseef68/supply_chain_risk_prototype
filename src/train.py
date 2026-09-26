import os
import glob
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

files = glob.glob("data/*.csv")

if not files:
    raise FileNotFoundError(
        "No CSV dataset found inside data/"
    )

file = files[0]

df = pd.read_csv(file)

print("=" * 70)
print("SUPPLY CHAIN DISRUPTION PREDICTION")
print("=" * 70)

print("\nDataset:", file)
print("Dataset shape:", df.shape)


# ============================================================
# DATE FEATURE ENGINEERING
# ============================================================

df["Date"] = pd.to_datetime(df["Date"])

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Day_of_Week"] = df["Date"].dt.dayofweek


# ============================================================
# REMOVE IDENTIFIERS
# ============================================================

df = df.drop(
    columns=["Shipment_ID", "Date"]
)


# ============================================================
# TARGET
# ============================================================

X = df.drop(
    columns=["Disruption_Occurred"]
)

y = df["Disruption_Occurred"]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(

    transformers=[

        (
            "numerical",
            StandardScaler(),
            numerical_features
        ),

        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        )

    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=RANDOM_STATE,

    stratify=y
)


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# MODELS
# ============================================================

models = {

    "Logistic Regression":

        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

    "Random Forest":

        RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1
        )
}


# ============================================================
# TRAINING
# ============================================================

results = []

trained_pipelines = {}


for name, model in models.items():

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)


    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    pipeline = Pipeline(

        steps=[

            (
                "preprocessing",
                preprocessor
            ),

            (
                "model",
                model
            )

        ]

    )


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    y_probability = pipeline.predict_proba(
        X_test
    )[:, 1]


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

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

    auc = roc_auc_score(
        y_test,
        y_probability
    )


    print(f"\nAccuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")


    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,

            target_names=[
                "No Disruption",
                "Disruption"
            ]
        )
    )


    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    results.append({

        "Model": name,

        "Accuracy": accuracy,

        "Precision": precision,

        "Recall": recall,

        "F1_Score": f1,

        "ROC_AUC": auc

    })


    trained_pipelines[name] = pipeline


    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_test,
        y_pred
    )


    disp = ConfusionMatrixDisplay(

        confusion_matrix=cm,

        display_labels=[
            "No Disruption",
            "Disruption"
        ]

    )


    disp.plot()

    plt.title(
        f"{name} - Confusion Matrix"
    )

    plt.tight_layout()


    filename = (
        name
        .lower()
        .replace(" ", "_")
    )


    plt.savefig(
        f"outputs/{filename}_confusion_matrix.png",
        dpi=300
    )

    plt.close()


# ============================================================
# MODEL COMPARISON
# ============================================================

results_df = pd.DataFrame(
    results
)


print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results_df.to_string(
        index=False
    )
)


results_df.to_csv(
    "outputs/model_comparison.csv",
    index=False
)


# ============================================================
# SELECT MODEL
# ============================================================

best_model_name = results_df.loc[
    results_df["ROC_AUC"].idxmax(),
    "Model"
]


best_pipeline = trained_pipelines[
    best_model_name
]


print(
    "\nSelected model:",
    best_model_name
)


# ============================================================
# SAVE COMPLETE PIPELINE
# ============================================================

joblib.dump(

    best_pipeline,

    "models/supply_chain_risk_model.joblib"

)


print(
    "\nSaved complete prediction pipeline:"
)

print(
    "models/supply_chain_risk_model.joblib"
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if best_model_name == "Random Forest":

    model = best_pipeline.named_steps[
        "model"
    ]

    preprocessing = best_pipeline.named_steps[
        "preprocessing"
    ]

    feature_names = (
        preprocessing
        .get_feature_names_out()
    )

    importance = pd.DataFrame({

        "Feature": feature_names,

        "Importance": model.feature_importances_

    })


    importance = importance.sort_values(

        "Importance",

        ascending=False

    )


    importance.to_csv(

        "outputs/feature_importance.csv",

        index=False

    )


    print(
        "\nTop 15 Important Features:"
    )

    print(
        importance.head(15).to_string(
            index=False
        )
    )


print(
    "\nTraining completed successfully."
)