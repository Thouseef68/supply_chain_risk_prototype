import os
import sys
import joblib
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "dataco_catboost_features_model.joblib"
)


# ============================================================
# MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

model = joblib.load(MODEL_PATH)


# ============================================================
# TRAINING FEATURE DEFINITIONS
# ============================================================

DROP_COLUMNS = [
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",

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
    "Product Status",

    "order date (DateOrders)"
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    df = df.copy()

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    date_col = "order date (DateOrders)"

    if date_col in df.columns:

        dt = pd.to_datetime(
            df[date_col],
            errors="coerce"
        )

        df["order_year"] = dt.dt.year
        df["order_month"] = dt.dt.month
        df["order_day"] = dt.dt.day
        df["order_dayofweek"] = dt.dt.dayofweek

        df["order_week"] = (
            dt.dt.isocalendar()
            .week
            .astype("float")
        )

    # --------------------------------------------------------
    # INTERACTION FEATURES
    # --------------------------------------------------------

    df["ShippingMode_ScheduledDays"] = (
        df["Shipping Mode"].astype(str)
        + "_"
        + df["Days for shipment (scheduled)"].astype(str)
    )

    df["ShippingMode_Market"] = (
        df["Shipping Mode"].astype(str)
        + "_"
        + df["Market"].astype(str)
    )

    df["ShippingMode_Region"] = (
        df["Shipping Mode"].astype(str)
        + "_"
        + df["Order Region"].astype(str)
    )

    df["ShippingMode_Month"] = (
        df["Shipping Mode"].astype(str)
        + "_"
        + df["order_month"].astype(str)
    )

    df["Market_ScheduledDays"] = (
        df["Market"].astype(str)
        + "_"
        + df["Days for shipment (scheduled)"].astype(str)
    )

    df["Market_Region"] = (
        df["Market"].astype(str)
        + "_"
        + df["Order Region"].astype(str)
    )

    df["Region_ScheduledDays"] = (
        df["Order Region"].astype(str)
        + "_"
        + df["Days for shipment (scheduled)"].astype(str)
    )

    return df


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_input(df):

    df = create_features(df)

    # Remove excluded columns
    columns_to_drop = [
        col
        for col in DROP_COLUMNS
        if col in df.columns
    ]

    df = df.drop(
        columns=columns_to_drop
    )

    # Exact model feature order
    model_features = list(
        model.feature_names_
    )

    missing = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing:

        raise ValueError(
            "Missing model features: "
            + ", ".join(missing)
        )

    df = df[
        model_features
    ].copy()

    # --------------------------------------------------------
    # CATEGORICAL / NUMERICAL FEATURES
    # --------------------------------------------------------

    categorical_indices = (
        model.get_cat_feature_indices()
    )

    categorical_features = [
        model_features[i]
        for i in categorical_indices
    ]

    numerical_features = [
        feature
        for feature in model_features
        if feature not in categorical_features
    ]

    # Categorical
    for col in categorical_features:

        df[col] = (
            df[col]
            .fillna("Unknown")
            .astype(str)
        )

    # Numerical
    for col in numerical_features:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        # CatBoost can handle NaN values
        # so don't calculate request-level
        # medians here.

    return df


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability >= 0.75:
        return "HIGH"

    if probability >= 0.50:
        return "MEDIUM"

    return "LOW"


# ============================================================
# RECOMMENDATIONS
# ============================================================

def generate_recommendation(
    probability,
    risk_level
):

    if risk_level == "HIGH":

        return {
            "priority": "Immediate attention",
            "message": (
                "This shipment shows a high predicted "
                "late-delivery risk. Review the delivery "
                "schedule, transportation planning and "
                "shipment monitoring requirements."
            ),
            "actions": [
                "Review planned delivery schedule",
                "Verify transportation availability",
                "Increase shipment monitoring",
                "Review route and regional conditions",
                "Consider alternative fulfillment options"
            ]
        }

    if risk_level == "MEDIUM":

        return {
            "priority": "Monitor closely",
            "message": (
                "This shipment shows a moderate predicted "
                "risk. Additional monitoring may help "
                "identify developing delivery problems."
            ),
            "actions": [
                "Review delivery schedule",
                "Monitor shipment status",
                "Check transportation availability"
            ]
        }

    return {
        "priority": "Normal monitoring",
        "message": (
            "The model does not identify a strong "
            "late-delivery risk signal for this shipment."
        ),
        "actions": [
            "Continue standard monitoring",
            "Maintain planned delivery schedule"
        ]
    }


# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_risk(input_data):

    # Dictionary → DataFrame
    if isinstance(input_data, dict):

        df = pd.DataFrame(
            [input_data]
        )

    elif isinstance(input_data, pd.DataFrame):

        df = input_data.copy()

    else:

        raise TypeError(
            "Input must be a dictionary "
            "or pandas DataFrame."
        )

    X = prepare_input(df)

    # Probability
    probabilities = (
        model.predict_proba(X)
    )

    probability = float(
        probabilities[0][1]
    )

    # Class
    prediction = int(
        model.predict(X)[0]
    )

    risk_level = classify_risk(
        probability
    )

    recommendation = (
        generate_recommendation(
            probability,
            risk_level
        )
    )

    return {

        "prediction": prediction,

        "risk_probability": round(
            probability,
            6
        ),

        "risk_percentage": round(
            probability * 100,
            2
        ),

        "risk_level": risk_level,

        "recommendation": recommendation
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

def get_model_info():

    return {

        "algorithm": "CatBoost Classifier",

        "features": len(
            model.feature_names_
        ),

        "feature_names": list(
            model.feature_names_
        ),

        "iterations": 5000,

        "learning_rate": 0.03,

        "depth": 9,

        "l2_regularization": 7,

        "accuracy": 0.9324,

        "precision": 0.9561,

        "recall": 0.9190,

        "f1_score": 0.9371,

        "roc_auc": 0.9870
    }