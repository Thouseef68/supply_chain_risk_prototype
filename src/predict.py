# src/predict.py

import os
import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "models/dataco_catboost_features_model.joblib"

# Columns removed during the verified training pipeline
DROP_COLUMNS = [
    # Leakage / post-outcome columns
    "Delivery Status",
    "Days for shipping (real)",
    "shipping date (DateOrders)",
    "Order Status",

    # Personal / identifier columns
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

    # Identifier/category IDs
    "Category Id",
    "Product Category Id",
    "Department Id",

    # Other removed columns
    "Order Zipcode",
    "Product Description",
    "Product Image",
    "Product Status",

    # Original date string after extracting date features
    "order date (DateOrders)",
]


ENGINEERED_FEATURES = [
    "ShippingMode_ScheduledDays",
    "ShippingMode_Market",
    "ShippingMode_Region",
    "ShippingMode_Month",
    "Market_ScheduledDays",
    "Market_Region",
    "Region_ScheduledDays",
]


def load_model():
    """
    Load the already-trained CatBoost model.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}\n"
            "Make sure you run this script from the project root."
        )

    model = joblib.load(MODEL_PATH)
    return model


def create_features(df):
    """
    Reproduce the feature engineering used for the verified model.
    """

    df = df.copy()

    # ---------------------------------------------------------
    # DATE FEATURES
    # ---------------------------------------------------------

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
        df["order_week"] = dt.dt.isocalendar().week.astype("float")

    # ---------------------------------------------------------
    # ENGINEERED INTERACTION FEATURES
    # ---------------------------------------------------------

    if "Shipping Mode" in df.columns and "Days for shipment (scheduled)" in df.columns:
        df["ShippingMode_ScheduledDays"] = (
            df["Shipping Mode"].astype(str)
            + "_"
            + df["Days for shipment (scheduled)"].astype(str)
        )

    if "Shipping Mode" in df.columns and "Market" in df.columns:
        df["ShippingMode_Market"] = (
            df["Shipping Mode"].astype(str)
            + "_"
            + df["Market"].astype(str)
        )

    if "Shipping Mode" in df.columns and "Order Region" in df.columns:
        df["ShippingMode_Region"] = (
            df["Shipping Mode"].astype(str)
            + "_"
            + df["Order Region"].astype(str)
        )

    if "Shipping Mode" in df.columns and "order_month" in df.columns:
        df["ShippingMode_Month"] = (
            df["Shipping Mode"].astype(str)
            + "_"
            + df["order_month"].astype(str)
        )

    if (
        "Market" in df.columns
        and "Days for shipment (scheduled)" in df.columns
    ):
        df["Market_ScheduledDays"] = (
            df["Market"].astype(str)
            + "_"
            + df["Days for shipment (scheduled)"].astype(str)
        )

    if "Market" in df.columns and "Order Region" in df.columns:
        df["Market_Region"] = (
            df["Market"].astype(str)
            + "_"
            + df["Order Region"].astype(str)
        )

    if (
        "Order Region" in df.columns
        and "Days for shipment (scheduled)" in df.columns
    ):
        df["Region_ScheduledDays"] = (
            df["Order Region"].astype(str)
            + "_"
            + df["Days for shipment (scheduled)"].astype(str)
        )

    return df


def prepare_input(df, model):
    """
    Prepare raw input so it matches the exact feature structure
    expected by the saved CatBoost model.
    """

    df = create_features(df)

    # Remove training-time excluded columns
    columns_to_drop = [
        c for c in DROP_COLUMNS
        if c in df.columns
    ]

    df = df.drop(columns=columns_to_drop)

    # ---------------------------------------------------------
    # MATCH MODEL FEATURE ORDER
    # ---------------------------------------------------------

    model_features = list(model.feature_names_)

    # Check for missing features
    missing_features = [
        feature
        for feature in model_features
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            "Input is missing model features:\n"
            + "\n".join(missing_features)
        )

    # Keep EXACT model feature order
    df = df[model_features].copy()

    # ---------------------------------------------------------
    # HANDLE MISSING VALUES
    # ---------------------------------------------------------

    cat_indices = model.get_cat_feature_indices()

    categorical_features = [
        model_features[i]
        for i in cat_indices
    ]

    numerical_features = [
        feature
        for feature in model_features
        if feature not in categorical_features
    ]

    # Categorical → string
    for col in categorical_features:
        df[col] = df[col].fillna("Unknown").astype(str)

    # Numerical → numeric
    for col in numerical_features:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        if df[col].isna().any():
            median_value = df[col].median()

            if pd.isna(median_value):
                median_value = 0

            df[col] = df[col].fillna(median_value)

    return df


def predict_risk(input_data):
    """
    Predict disruption / late-delivery risk.

    Parameters
    ----------
    input_data : dict or pandas.DataFrame

    Returns
    -------
    dict
    """

    model = load_model()

    # Convert dictionary to DataFrame
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data])

    elif isinstance(input_data, pd.DataFrame):
        df = input_data.copy()

    else:
        raise TypeError(
            "input_data must be a dictionary or pandas DataFrame."
        )

    X = prepare_input(df, model)

    # Probability
    probabilities = model.predict_proba(X)

    risk_probability = float(probabilities[0][1])

    # Class prediction
    prediction = int(
        model.predict(X)[0]
    )

    # ---------------------------------------------------------
    # BUSINESS RISK LEVEL
    # ---------------------------------------------------------

    if risk_probability >= 0.75:
        risk_level = "HIGH"

    elif risk_probability >= 0.50:
        risk_level = "MEDIUM"

    else:
        risk_level = "LOW"

    # ---------------------------------------------------------
    # RECOMMENDATION
    # ---------------------------------------------------------

    if risk_level == "HIGH":

        recommendation = (
            "Prioritize this shipment for immediate monitoring. "
            "Review transportation conditions, route planning, "
            "and fulfillment scheduling."
        )

    elif risk_level == "MEDIUM":

        recommendation = (
            "Monitor this shipment closely and review the "
            "planned delivery schedule."
        )

    else:

        recommendation = (
            "Normal shipment monitoring is recommended."
        )

    return {
        "prediction": prediction,
        "risk_probability": risk_probability,
        "risk_percentage": risk_probability * 100,
        "risk_level": risk_level,
        "recommendation": recommendation,
    }


# -------------------------------------------------------------
# TEST
# -------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("SUPPLY CHAIN RISK PREDICTION TEST")
    print("=" * 60)

    model = load_model()

    print("\nModel loaded successfully.")

    print(
        f"Number of model features: "
        f"{len(model.feature_names_)}"
    )

    print("\nModel features:")

    for i, feature in enumerate(model.feature_names_, 1):
        print(f"{i:02d}. {feature}")

    print("\nPrediction module ready.")