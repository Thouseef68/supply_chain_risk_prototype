import os
import joblib
import pandas as pd

from predict import predict_risk


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DISRUPTION_MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "models",
    "disruption_catboost_model.joblib"
)

disruption_model = joblib.load(
    DISRUPTION_MODEL_PATH
)


DISRUPTION_CATEGORICAL = [
    "Origin_Port",
    "Destination_Port",
    "Transport_Mode",
    "Product_Category",
    "Weather_Condition",
]


DISRUPTION_FEATURES = [
    "Origin_Port",
    "Destination_Port",
    "Transport_Mode",
    "Product_Category",
    "Distance_km",
    "Weight_MT",
    "Fuel_Price_Index",
    "Geopolitical_Risk_Score",
    "Weather_Condition",
    "Carrier_Reliability_Score",
    "Lead_Time_Days",
    "date_year",
    "date_month",
    "date_day",
    "date_dayofweek",
    "date_week",
]


def detect_schema(columns):

    columns = set(columns)

    disruption_signature = {
        "Origin_Port",
        "Destination_Port",
        "Transport_Mode",
        "Distance_km",
        "Geopolitical_Risk_Score",
        "Carrier_Reliability_Score",
    }

    dataco_signature = {
        "Shipping Mode",
        "Market",
        "Order Region",
        "Customer Segment",
        "Days for shipment (scheduled)",
    }

    disruption_matches = len(
        disruption_signature.intersection(columns)
    )

    dataco_matches = len(
        dataco_signature.intersection(columns)
    )

    if disruption_matches >= 4:
        return "disruption"

    if dataco_matches >= 3:
        return "dataco"

    return "unknown"


def disruption_probability(df):

    data = df.copy()

    # Date features
    if "Date" in data.columns:

        dt = pd.to_datetime(
            data["Date"],
            errors="coerce"
        )

    else:

        dt = pd.Series(
            pd.NaT,
            index=data.index
        )

    data["date_year"] = dt.dt.year
    data["date_month"] = dt.dt.month
    data["date_day"] = dt.dt.day
    data["date_dayofweek"] = dt.dt.dayofweek
    data["date_week"] = (
        dt.dt.isocalendar()
        .week
        .astype(float)
    )

    # Keep exactly the model features
    X = data[
        DISRUPTION_FEATURES
    ].copy()

    # Categorical values
    for column in DISRUPTION_CATEGORICAL:

        X[column] = (
            X[column]
            .fillna("Unknown")
            .astype(str)
        )

    # Numerical values
    for column in X.columns:

        if column not in DISRUPTION_CATEGORICAL:

            X[column] = pd.to_numeric(
                X[column],
                errors="coerce"
            )

    probabilities = (
        disruption_model
        .predict_proba(X)[:, 1]
    )

    return probabilities


def probability_to_level(probability):

    if probability >= 0.70:
        return "HIGH"

    if probability >= 0.50:
        return "MEDIUM"

    return "LOW"


def predict_batch_hybrid(df):

    schema = detect_schema(
        df.columns
    )

    if schema == "unknown":

        raise ValueError(
            "Unsupported CSV format. "
            "The file does not match the "
            "DataCo or Supply Chain Disruption schema."
        )

    if schema == "dataco":

        results = []

        for index, row in df.iterrows():

            try:

                result = predict_risk(
                    row.to_dict()
                )

                results.append({
                    "row_number": index + 1,
                    "model": "DataCo CatBoost",
                    "risk_type": "Late Delivery",
                    "prediction": result["prediction"],
                    "risk_probability": result[
                        "risk_probability"
                    ],
                    "risk_percentage": result[
                        "risk_percentage"
                    ],
                    "risk_level": result[
                        "risk_level"
                    ],
                })

            except Exception as error:

                results.append({
                    "row_number": index + 1,
                    "model": "DataCo CatBoost",
                    "risk_type": "Late Delivery",
                    "prediction": None,
                    "risk_probability": None,
                    "risk_percentage": None,
                    "risk_level": "ERROR",
                    "error": str(error),
                })

        return results, schema

    # --------------------------------
    # DISRUPTION MODEL
    # --------------------------------

    probabilities = disruption_probability(
        df
    )

    results = []

    for index, probability in enumerate(
        probabilities
    ):

        probability = float(
            probability
        )

        prediction = int(
            probability >= 0.50
        )

        results.append({
            "row_number": index + 1,
            "model": "Disruption CatBoost",
            "risk_type": "Supply Chain Disruption",
            "prediction": prediction,
            "risk_probability": round(
                probability,
                6
            ),
            "risk_percentage": round(
                probability * 100,
                2
            ),
            "risk_level":
                probability_to_level(
                    probability
                ),
        })

    return results, schema