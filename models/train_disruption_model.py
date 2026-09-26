import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from catboost import CatBoostClassifier

DATA_PATH = "supply_chain_risk_results.csv"
MODEL_PATH = "models/disruption_catboost_model.joblib"
META_PATH = "models/disruption_model_metadata.json"

df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)

target = "Disruption_Occurred"

drop_cols = [
    "Shipment_ID",
    "Date",
    target,
    "Prediction",
    "Risk Probability",
    "Risk Percentage",
    "Risk Level",
]

dt = pd.to_datetime(df["Date"], errors="coerce")
df["date_year"] = dt.dt.year
df["date_month"] = dt.dt.month
df["date_day"] = dt.dt.day
df["date_dayofweek"] = dt.dt.dayofweek
df["date_week"] = dt.dt.isocalendar().week.astype(float)

X = df.drop(columns=drop_cols)
y = df[target].astype(int)

categorical_features = [
    c for c in X.columns if X[c].dtype == "object"
]

categorical_indices = [
    X.columns.get_loc(c) for c in categorical_features
]

for c in categorical_features:
    X[c] = X[c].fillna("Unknown").astype(str)

for c in X.columns:
    if c not in categorical_features:
        X[c] = pd.to_numeric(X[c], errors="coerce")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = CatBoostClassifier(
    iterations=1500,
    learning_rate=0.03,
    depth=7,
    l2_leaf_reg=7,
    loss_function="Logloss",
    eval_metric="AUC",
    random_seed=42,
    od_type="Iter",
    od_wait=100,
    use_best_model=True,
    verbose=False,
    thread_count=-1,
)

model.fit(
    X_train,
    y_train,
    cat_features=categorical_indices,
    eval_set=(X_test, y_test),
    verbose=False,
)

probability = model.predict_proba(X_test)[:, 1]
prediction = (probability >= 0.5).astype(int)

metrics = {
    "accuracy": round(accuracy_score(y_test, prediction), 6),
    "precision": round(precision_score(y_test, prediction, zero_division=0), 6),
    "recall": round(recall_score(y_test, prediction, zero_division=0), 6),
    "f1": round(f1_score(y_test, prediction, zero_division=0), 6),
    "roc_auc": round(roc_auc_score(y_test, probability), 6),
    "test_size": len(y_test),
    "best_iteration": int(model.get_best_iteration()),
}

Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)

joblib.dump(model, MODEL_PATH)

metadata = {
    "model_name": "Disruption CatBoost",
    "target": target,
    "features": X.columns.tolist(),
    "categorical_features": categorical_features,
    "metrics": metrics,
    "confusion_matrix": confusion_matrix(
        y_test, prediction
    ).tolist(),
    "training_rows": len(X_train),
    "test_rows": len(X_test),
    "positive_rate": float(y.mean()),
}

with open(META_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Model saved:", MODEL_PATH)
print(json.dumps(metrics, indent=2))
