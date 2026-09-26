import os
import json
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any
from datetime import datetime
from functools import lru_cache
from io import BytesIO

from predict import (
    predict_risk,
    get_model_info
)

from hybrid_predict import (
    predict_batch_hybrid,
    disruption_probability,
    detect_schema
)

from database import (
    init_database,
    save_prediction,
    get_prediction_history,
    clear_prediction_history
)

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataco",
    "DataCoSupplyChainDataset.csv"
)


@lru_cache(maxsize=1)
def load_dashboard_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Dataset not found")

    df = pd.read_csv(
        DATA_PATH,
        encoding="latin1",
        low_memory=False
    )

    return df


# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SupplyChainIQ API",
    description=(
        "AI-Powered Supply Chain Risk "
        "Intelligence & Prediction API"
    ),
    version="1.0.0"
)


init_database()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,

    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"]
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class PredictionRequest(BaseModel):

    data: dict[str, Any]


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {

        "status": "online",

        "service": "SupplyChainIQ API",

        "model": "CatBoost",

        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/api/model")
def model_information():

    return get_model_info()


# ============================================================
# PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(request: PredictionRequest):

    try:

        result = predict_risk(
            request.data
        )

        save_prediction(
            model="DataCo CatBoost",
            risk_type="Late Delivery",
            prediction=result["prediction"],
            risk_probability=result["risk_probability"],
            risk_percentage=result["risk_percentage"],
            risk_level=result["risk_level"],
            input_data=json.dumps(request.data)
        )

        return {

            "success": True,

            "result": result

        }

    except Exception as error:

        return {

            "success": False,

            "error": str(error)

        }


# ============================================================
# DASHBOARD ANALYTICS
# ============================================================

@app.get("/api/dashboard")
def dashboard():
    try:
        df = load_dashboard_data()

        total_orders = len(df)

        risk_cases = int(
            df["Late_delivery_risk"].sum()
        )

        no_risk_cases = total_orders - risk_cases

        risk_rate = (
            risk_cases / total_orders * 100
        )

        # Shipping Mode
        shipping_risk = (
            df.groupby("Shipping Mode")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        shipping_risk["risk_percentage"] = (
            shipping_risk["Late_delivery_risk"] * 100
        ).round(2)

        shipping_risk = shipping_risk.drop(
            columns=["Late_delivery_risk"]
        )

        # Market
        market_risk = (
            df.groupby("Market")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        market_risk["risk_percentage"] = (
            market_risk["Late_delivery_risk"] * 100
        ).round(2)

        market_risk = market_risk.drop(
            columns=["Late_delivery_risk"]
        )

        # Customer Segment
        segment_risk = (
            df.groupby("Customer Segment")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        segment_risk["risk_percentage"] = (
            segment_risk["Late_delivery_risk"] * 100
        ).round(2)

        segment_risk = segment_risk.drop(
            columns=["Late_delivery_risk"]
        )

        return {
            "success": True,

            "summary": {
                "total_orders": total_orders,
                "risk_cases": risk_cases,
                "no_risk_cases": no_risk_cases,
                "risk_rate": round(risk_rate, 2),
                "model_accuracy": 93.24,
                "model_roc_auc": 98.70
            },

            "shipping_mode":
                shipping_risk.to_dict(orient="records"),

            "market":
                market_risk.to_dict(orient="records"),

            "customer_segment":
                segment_risk.to_dict(orient="records")
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# ANALYTICS
# ============================================================

@app.get("/api/analytics")
def analytics():
    try:
        df = load_dashboard_data()

        # Create calendar features from the original order date
        if "order date (DateOrders)" in df.columns:
            df["order date (DateOrders)"] = pd.to_datetime(
                df["order date (DateOrders)"],
                errors="coerce"
            )

            df["order_month"] = (
                df["order date (DateOrders)"].dt.month
            )
        else:
            raise ValueError(
                "Order date column not found in DataCo dataset"
            )

        # Risk by Region
        region_risk = (
            df.groupby("Order Region")["Late_delivery_risk"]
            .agg(["mean", "count"])
            .reset_index()
        )

        region_risk["risk_percentage"] = (
            region_risk["mean"] * 100
        ).round(2)

        region_risk = region_risk.rename(
            columns={
                "Order Region": "region",
                "count": "orders"
            }
        )

        region_risk = region_risk[
            ["region", "orders", "risk_percentage"]
        ]

        # Risk by Transaction Type
        type_risk = (
            df.groupby("Type")["Late_delivery_risk"]
            .agg(["mean", "count"])
            .reset_index()
        )

        type_risk["risk_percentage"] = (
            type_risk["mean"] * 100
        ).round(2)

        type_risk = type_risk.rename(
            columns={
                "Type": "type",
                "count": "orders"
            }
        )

        type_risk = type_risk[
            ["type", "orders", "risk_percentage"]
        ]

        # Monthly Risk
        monthly_risk = (
            df.groupby("order_month")["Late_delivery_risk"]
            .agg(["mean", "count"])
            .reset_index()
        )

        monthly_risk["risk_percentage"] = (
            monthly_risk["mean"] * 100
        ).round(2)

        monthly_risk = monthly_risk.rename(
            columns={
                "order_month": "month",
                "count": "orders"
            }
        )

        monthly_risk = monthly_risk[
            ["month", "orders", "risk_percentage"]
        ].sort_values("month")

        # Market x Region heatmap
        heatmap = (
            df.groupby(
                ["Market", "Order Region"]
            )["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        heatmap["risk_percentage"] = (
            heatmap["Late_delivery_risk"] * 100
        ).round(2)

        heatmap = heatmap.rename(
            columns={
                "Market": "market",
                "Order Region": "region"
            }
        )

        heatmap = heatmap[
            ["market", "region", "risk_percentage"]
        ]

        return {
            "success": True,
            "region_risk": region_risk.to_dict(
                orient="records"
            ),
            "type_risk": type_risk.to_dict(
                orient="records"
            ),
            "monthly_risk": monthly_risk.to_dict(
                orient="records"
            ),
            "heatmap": heatmap.to_dict(
                orient="records"
            )
        }
    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# WHAT-IF SCENARIO
# ============================================================

@app.post("/api/what-if")
def what_if(request: PredictionRequest):
    try:
        data = request.data

        schema = detect_schema(
            data.keys()
        )

        if schema == "unknown":
            return {
                "success": False,
                "error": (
                    "Unable to determine the shipment "
                    "schema."
                )
            }

        if schema == "dataco":

            result = predict_risk(data)

            save_prediction(
                model="DataCo CatBoost",
                risk_type="Late Delivery",
                prediction=result["prediction"],
                risk_probability=result["risk_probability"],
                risk_percentage=result["risk_percentage"],
                risk_level=result["risk_level"],
                input_data=json.dumps(data)
            )

            return {
                "success": True,
                "model": "DataCo CatBoost",
                "risk_type": "Late Delivery",
                "result": result
            }

        # Disruption model
        df = pd.DataFrame([data])

        probabilities = disruption_probability(df)

        probability = float(
            probabilities[0]
        )

        prediction = int(
            probability >= 0.50
        )

        if probability >= 0.70:
            level = "HIGH"
        elif probability >= 0.50:
            level = "MEDIUM"
        else:
            level = "LOW"

        save_prediction(
            model="Disruption CatBoost",
            risk_type="Supply Chain Disruption",
            prediction=prediction,
            risk_probability=probability,
            risk_percentage=probability * 100,
            risk_level=level,
            input_data=json.dumps(data)
        )

        return {
            "success": True,
            "model": "Disruption CatBoost",
            "risk_type": "Supply Chain Disruption",
            "result": {
                "prediction": prediction,
                "risk_probability": round(
                    probability,
                    6
                ),
                "risk_percentage": round(
                    probability * 100,
                    2
                ),
                "risk_level": level
            }
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/api/batch-predict")
async def batch_predict(
    file: UploadFile = File(...)
):

    try:

        if not file.filename.lower().endswith(".csv"):

            return {
                "success": False,
                "error": "Only CSV files are supported."
            }

        contents = await file.read()

        if not contents:

            return {
                "success": False,
                "error": "The uploaded CSV is empty."
            }

        df = pd.read_csv(
            BytesIO(contents),
            encoding="latin1",
            low_memory=False
        )

        if len(df) == 0:

            return {
                "success": False,
                "error": "The uploaded CSV contains no rows."
            }

        # Maximum batch size
        if len(df) > 6000:

            return {
                "success": False,
                "error": "Maximum 6,000 shipments per batch."
            }

        # Automatically select model
        results, schema = predict_batch_hybrid(
            df
        )

        # Summary
        high = sum(
            1 for r in results
            if r["risk_level"] == "HIGH"
        )

        medium = sum(
            1 for r in results
            if r["risk_level"] == "MEDIUM"
        )

        low = sum(
            1 for r in results
            if r["risk_level"] == "LOW"
        )

        errors = sum(
            1 for r in results
            if r["risk_level"] == "ERROR"
        )

        output = df.copy()

        output["Model"] = [
            r["model"]
            for r in results
        ]

        output["Risk Type"] = [
            r["risk_type"]
            for r in results
        ]

        output["Prediction"] = [
            r["prediction"]
            for r in results
        ]

        output["Risk Probability"] = [
            r["risk_probability"]
            for r in results
        ]

        output["Risk Percentage"] = [
            r["risk_percentage"]
            for r in results
        ]

        output["Risk Level"] = [
            r["risk_level"]
            for r in results
        ]

        return {
            "success": True,

            "filename": file.filename,

            "dataset_type": schema,

            "model": (
                "DataCo CatBoost"
                if schema == "dataco"
                else "Disruption CatBoost"
            ),

            "risk_type": (
                "Late Delivery"
                if schema == "dataco"
                else "Supply Chain Disruption"
            ),

            "total_rows": len(results),

            "high_risk": high,

            "medium_risk": medium,

            "low_risk": low,

            "errors": errors,

            "results": results,

            "data":
                output.to_dict(
                    orient="records"
                )
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# BATCH TEMPLATE
# ============================================================

@app.get("/api/batch-template")
def batch_template():
    try:
        df = load_dashboard_data()

        sample = df.sample(
            n=min(25, len(df)),
            random_state=42
        )

        csv_data = sample.to_csv(
            index=False
        )

        return {
            "success": True,
            "filename": "supply_chain_batch_template.csv",
            "csv": csv_data
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "name": "SupplyChainIQ",

        "message": (
            "AI-Powered Supply Chain "
            "Risk Intelligence API"
        ),

        "documentation": "/docs",

        "health": "/api/health"
    }


# ============================================================
# SHIPMENT EXPLORER
# ============================================================

@app.get("/api/shipments")
def get_shipments(
    search: str = "",
    risk: str = "ALL",
    shipping_mode: str = "ALL",
    market: str = "ALL",
    page: int = 1,
    limit: int = 25
):
    try:
        df = load_dashboard_data()

        # Safety limits
        page = max(1, page)
        limit = min(max(1, limit), 100)

        filtered = df

        # Search
        if search.strip():
            query = search.strip().lower()

            searchable_columns = [
                "Order City",
                "Order State",
                "Order Country",
                "Customer City",
                "Customer State",
                "Customer Country",
                "Product Name",
                "Order Region"
            ]

            mask = pd.Series(
                False,
                index=filtered.index
            )

            for column in searchable_columns:
                if column in filtered.columns:
                    mask = mask | (
                        filtered[column]
                        .fillna("")
                        .astype(str)
                        .str.lower()
                        .str.contains(
                            query,
                            regex=False
                        )
                    )

            filtered = filtered[mask]

        # Shipping mode
        if shipping_mode != "ALL":
            filtered = filtered[
                filtered["Shipping Mode"]
                .astype(str)
                == shipping_mode
            ]

        # Market
        if market != "ALL":
            filtered = filtered[
                filtered["Market"]
                .astype(str)
                == market
            ]

        # Risk
        if risk != "ALL":
            if risk == "HIGH":
                filtered = filtered[
                    filtered["Late_delivery_risk"] == 1
                ]

            elif risk == "LOW":
                filtered = filtered[
                    filtered["Late_delivery_risk"] == 0
                ]

        total = len(filtered)

        start = (page - 1) * limit
        end = start + limit

        page_data = filtered.iloc[
            start:end
        ].copy()

        shipments = []

        for index, row in page_data.iterrows():

            shipment_id = (
                row.get("Order Id", None)
            )

            shipments.append({
                "id": (
                    str(shipment_id)
                    if pd.notna(shipment_id)
                    else str(index)
                ),

                "order_city": str(
                    row.get("Order City", "")
                ),

                "order_state": str(
                    row.get("Order State", "")
                ),

                "order_country": str(
                    row.get("Order Country", "")
                ),

                "customer_city": str(
                    row.get("Customer City", "")
                ),

                "customer_country": str(
                    row.get("Customer Country", "")
                ),

                "market": str(
                    row.get("Market", "")
                ),

                "region": str(
                    row.get("Order Region", "")
                ),

                "shipping_mode": str(
                    row.get("Shipping Mode", "")
                ),

                "scheduled_days": (
                    float(
                        row["Days for shipment (scheduled)"]
                    )
                    if pd.notna(
                        row.get(
                            "Days for shipment (scheduled)"
                        )
                    )
                    else None
                ),

                "sales": (
                    round(
                        float(row["Sales"]),
                        2
                    )
                    if pd.notna(
                        row.get("Sales")
                    )
                    else None
                ),

                "quantity": (
                    int(row["Order Item Quantity"])
                    if pd.notna(
                        row.get(
                            "Order Item Quantity"
                        )
                    )
                    else None
                ),

                "risk": (
                    "HIGH"
                    if int(
                        row["Late_delivery_risk"]
                    ) == 1
                    else "LOW"
                )
            })

        return {
            "success": True,
            "total": total,
            "page": page,
            "limit": limit,
            "total_pages": (
                (total + limit - 1) // limit
            ),
            "shipments": shipments
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


@app.get("/api/shipment-filters")
def shipment_filters():
    try:
        df = load_dashboard_data()

        return {
            "success": True,

            "shipping_modes": sorted(
                df["Shipping Mode"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            ),

            "markets": sorted(
                df["Market"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

@app.get("/api/model-intelligence")
def model_intelligence():
    try:
        from hybrid_predict import disruption_model

        dataco_model_info = get_model_info()

        disruption_metrics = {
            "accuracy": 73.40,
            "precision": 78.58,
            "recall": 77.81,
            "f1": 78.20,
            "roc_auc": 82.91,
            "test_samples": 1000
        }

        disruption_features = [
            {
                "feature": "Geopolitical Risk Score",
                "importance": 0
            },
            {
                "feature": "Carrier Reliability Score",
                "importance": 0
            },
            {
                "feature": "Lead Time Days",
                "importance": 0
            },
            {
                "feature": "Distance km",
                "importance": 0
            },
            {
                "feature": "Weather Condition",
                "importance": 0
            },
            {
                "feature": "Transport Mode",
                "importance": 0
            },
            {
                "feature": "Fuel Price Index",
                "importance": 0
            },
            {
                "feature": "Weight MT",
                "importance": 0
            },
            {
                "feature": "Product Category",
                "importance": 0
            },
            {
                "feature": "Origin Port",
                "importance": 0
            },
            {
                "feature": "Destination Port",
                "importance": 0
            }
        ]

        # Actual CatBoost feature importance
        importances = (
            disruption_model
            .get_feature_importance()
        )

        feature_names = (
            disruption_model.feature_names_
        )

        disruption_features = sorted(
            [
                {
                    "feature": name,
                    "importance": round(
                        float(importance),
                        3
                    )
                }
                for name, importance
                in zip(
                    feature_names,
                    importances
                )
            ],
            key=lambda x: x["importance"],
            reverse=True
        )

        return {
            "success": True,

            "models": {

                "dataco": {
                    "name": "DataCo CatBoost",
                    "purpose": "Late Delivery Risk",
                    "metrics": {
                        "accuracy": 93.24,
                        "precision": 95.61,
                        "recall": 91.90,
                        "f1": 93.71,
                        "roc_auc": 98.70,
                        "test_samples": 27078
                    },
                    "features": [
                        {
                            "feature": "Order City",
                            "importance": 10.79
                        },
                        {
                            "feature": "Type",
                            "importance": 10.72
                        },
                        {
                            "feature": "Customer City",
                            "importance": 8.94
                        },
                        {
                            "feature": "ShippingMode_Month",
                            "importance": 7.37
                        },
                        {
                            "feature": "Latitude",
                            "importance": 6.27
                        },
                        {
                            "feature": "Customer State",
                            "importance": 4.55
                        },
                        {
                            "feature": "Shipping Mode",
                            "importance": 4.36
                        },
                        {
                            "feature": "Longitude",
                            "importance": 3.85
                        },
                        {
                            "feature": "ShippingMode_Region",
                            "importance": 3.82
                        },
                        {
                            "feature": "ShippingMode_Market",
                            "importance": 3.69
                        }
                    ]
                },

                "disruption": {
                    "name": "Disruption CatBoost",
                    "purpose": "Supply Chain Disruption",
                    "metrics": disruption_metrics,
                    "features": disruption_features
                }
            },

            "routing": {
                "dataco": "DataCo shipment schema",
                "disruption": "Supply chain disruption schema"
            }
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# PREDICTION HISTORY
# ============================================================

@app.get("/api/prediction-history")
def prediction_history(limit: int = 100):

    try:

        limit = max(1, min(limit, 500))

        records = get_prediction_history(limit)

        return {
            "success": True,
            "count": len(records),
            "records": records
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


@app.delete("/api/prediction-history")
def delete_prediction_history():

    try:

        clear_prediction_history()

        return {
            "success": True,
            "message": "Prediction history cleared."
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }


# ============================================================
# HISTORY SUMMARY
# ============================================================

@app.get("/api/history-summary")
def history_summary():

    try:
        records = get_prediction_history(500)

        total = len(records)

        high = sum(
            1 for r in records
            if r["risk_level"] == "HIGH"
        )

        medium = sum(
            1 for r in records
            if r["risk_level"] == "MEDIUM"
        )

        low = sum(
            1 for r in records
            if r["risk_level"] == "LOW"
        )

        models = {}

        for record in records:
            model = record["model"]
            models[model] = models.get(model, 0) + 1

        latest = records[0] if records else None

        return {
            "success": True,
            "total_predictions": total,
            "high_risk": high,
            "medium_risk": medium,
            "low_risk": low,
            "models": models,
            "latest": latest
        }

    except Exception as error:

        return {
            "success": False,
            "error": str(error)
        }