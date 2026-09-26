import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
import plotly.express as px


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supply Chain Risk Intelligence",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "models/dataco_catboost_features_model.joblib"
DATA_PATH = "data/dataco/DataCoSupplyChainDataset.csv"


# ============================================================
# LOAD PREDICTION MODULE
# ============================================================

sys.path.append("src")

from predict import predict_risk, prepare_input, load_model


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .risk-high {
        background: #fee2e2;
        color: #991b1b;
        padding: 15px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }

    .risk-medium {
        background: #fef3c7;
        color: #92400e;
        padding: 15px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }

    .risk-low {
        background: #dcfce7;
        color: #166534;
        padding: 15px;
        border-radius: 10px;
        font-size: 24px;
        font-weight: bold;
        text-align: center;
    }

    .section-title {
        font-size: 26px;
        font-weight: 700;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def get_model():

    if not os.path.exists(MODEL_PATH):
        return None

    return load_model()


model = get_model()


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():

    if not os.path.exists(DATA_PATH):
        return None

    return pd.read_csv(
        DATA_PATH,
        encoding="latin1"
    )


data = load_data()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🚚 Supply Chain AI")

st.sidebar.markdown(
    """
    ### Risk Intelligence System

    Predict potential **late-delivery / supply-chain risk**
    using a trained CatBoost machine-learning model.

    ---
    """
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Executive Dashboard",
        "Risk Prediction",
        "Batch Analysis",
        "Model Intelligence"
    ]
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Data Source: DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS"
)

st.sidebar.caption(
    "Model: CatBoost Classifier"
)


# ============================================================
# MODEL CHECK
# ============================================================

if model is None:

    st.error(
        "Model file not found. "
        "Make sure models/dataco_catboost_features_model.joblib exists."
    )

    st.stop()


# ============================================================
# EXECUTIVE DASHBOARD
# ============================================================

if page == "Executive Dashboard":

    st.title("Supply Chain Risk Intelligence")
    st.markdown(
        "### Predictive analytics for shipment disruption and late-delivery risk"
    )

    st.markdown("---")

    if data is not None:

        total_orders = len(data)

        high_risk_count = int(
            (data["Late_delivery_risk"] == 1).sum()
        )

        risk_rate = (
            high_risk_count / total_orders * 100
        )

    else:

        total_orders = 180519
        high_risk_count = 98977
        risk_rate = 54.83


    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Orders Analyzed",
            f"{total_orders:,}"
        )

    with col2:

        st.metric(
            "Risk Cases",
            f"{high_risk_count:,}"
        )

    with col3:

        st.metric(
            "Historical Risk Rate",
            f"{risk_rate:.2f}%"
        )

    with col4:

        st.metric(
            "Model ROC-AUC",
            "98.70%"
        )


    st.markdown("---")


    # --------------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">System Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            """
            **What this system does**

            The system analyzes historical supply-chain order
            characteristics and estimates the probability that
            a shipment will experience late-delivery risk.

            The prediction engine uses:

            • CatBoost classification  
            • Geographic information  
            • Shipping characteristics  
            • Customer information  
            • Order information  
            • Temporal features  
            • Engineered interaction features
            """
        )

    with col2:

        st.success(
            """
            **Business value**

            The prediction can help supply-chain teams:

            • Identify high-risk shipments  
            • Prioritize monitoring  
            • Review transportation planning  
            • Detect operational risk patterns  
            • Support proactive decision-making
            """
        )


    # --------------------------------------------------------
    # SHIPPING MODE ANALYSIS
    # --------------------------------------------------------

    if data is not None:

        st.markdown(
            '<div class="section-title">Risk by Shipping Mode</div>',
            unsafe_allow_html=True
        )

        mode_risk = (
            data.groupby("Shipping Mode")["Late_delivery_risk"]
            .mean()
            .reset_index()
        )

        mode_risk["Risk %"] = (
            mode_risk["Late_delivery_risk"] * 100
        )

        fig = px.bar(
            mode_risk,
            x="Shipping Mode",
            y="Risk %",
            text="Risk %",
            title="Historical Late-Delivery Risk by Shipping Mode"
        )

        fig.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside"
        )

        fig.update_layout(
            yaxis_title="Risk (%)",
            xaxis_title=""
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


# ============================================================
# RISK PREDICTION
# ============================================================

elif page == "Risk Prediction":

    st.title("🎯 Shipment Risk Prediction")

    st.markdown(
        "Enter shipment characteristics to estimate late-delivery risk."
    )

    st.markdown("---")


    # --------------------------------------------------------
    # DEMO SAMPLE
    # --------------------------------------------------------

    if data is not None:

        st.subheader("Quick Demo")

        if st.button("Load Real DataCo Sample"):

            sample = data.sample(
                1,
                random_state=42
            ).iloc[0]

            st.session_state["demo_sample"] = sample.to_dict()


    sample_data = st.session_state.get(
        "demo_sample",
        {}
    )


    # --------------------------------------------------------
    # BASIC INPUTS
    # --------------------------------------------------------

    st.subheader("Shipment Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        shipping_mode = st.selectbox(
            "Shipping Mode",
            [
                "Standard Class",
                "Second Class",
                "First Class",
                "Same Day"
            ],
            index=0
        )

    with col2:

        scheduled_days = st.selectbox(
            "Scheduled Shipping Days",
            [0, 1, 2, 4],
            index=3
        )

    with col3:

        market = st.selectbox(
            "Market",
            [
                "USCA",
                "LATAM",
                "Europe",
                "Pacific Asia",
                "Africa"
            ]
        )


    col1, col2, col3 = st.columns(3)

    with col1:

        order_region = st.text_input(
            "Order Region",
            value="Western Europe"
        )

    with col2:

        customer_segment = st.selectbox(
            "Customer Segment",
            [
                "Consumer",
                "Corporate",
                "Home Office"
            ]
        )

    with col3:

        order_country = st.text_input(
            "Order Country",
            value="United States"
        )


    # --------------------------------------------------------
    # ORDER VALUE
    # --------------------------------------------------------

    st.subheader("Order Information")

    col1, col2, col3 = st.columns(3)

    with col1:

        sales = st.number_input(
            "Sales",
            min_value=0.0,
            value=100.0
        )

    with col2:

        product_price = st.number_input(
            "Product Price",
            min_value=0.0,
            value=100.0
        )

    with col3:

        quantity = st.number_input(
            "Order Quantity",
            min_value=1,
            value=1
        )


    # --------------------------------------------------------
    # LOCATION
    # --------------------------------------------------------

    st.subheader("Location")

    col1, col2, col3 = st.columns(3)

    with col1:

        customer_city = st.text_input(
            "Customer City",
            value="New York City"
        )

    with col2:

        customer_state = st.text_input(
            "Customer State",
            value="NY"
        )

    with col3:

        order_city = st.text_input(
            "Order City",
            value="New York City"
        )


    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    st.subheader("Order Date")

    order_date = st.date_input(
        "Order Date"
    )


    st.markdown("---")


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🚀 Predict Supply Chain Risk",
        type="primary",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # Construct input record
            # ------------------------------------------------

            input_data = {

                "Type": "DEBIT",

                "Days for shipment (scheduled)": scheduled_days,

                "Benefit per order": 50.0,

                "Sales per customer": sales,

                "Category Name": "Unknown",

                "Customer City": customer_city,

                "Customer Country": "United States",

                "Customer Segment": customer_segment,

                "Customer State": customer_state,

                "Department Name": "Unknown",

                "Latitude": 40.7128,

                "Longitude": -74.0060,

                "Market": market,

                "Order City": order_city,

                "Order Country": order_country,

                "Order Item Discount": 0.0,

                "Order Item Discount Rate": 0.0,

                "Order Item Product Price": product_price,

                "Order Item Profit Ratio": 0.20,

                "Order Item Quantity": quantity,

                "Sales": sales,

                "Order Item Total": sales,

                "Order Profit Per Order": sales * 0.20,

                "Order Region": order_region,

                "Order State": customer_state,

                "Product Name": "Unknown",

                "Product Price": product_price,

                "Shipping Mode": shipping_mode,

                "order date (DateOrders)": str(order_date)

            }


            result = predict_risk(
                input_data
            )


            # ------------------------------------------------
            # DISPLAY RESULT
            # ------------------------------------------------

            st.markdown("---")

            st.subheader("Prediction Result")


            probability = result[
                "risk_percentage"
            ]

            risk_level = result[
                "risk_level"
            ]


            if risk_level == "HIGH":

                st.markdown(
                    '<div class="risk-high">'
                    f'HIGH RISK — {probability:.2f}%'
                    '</div>',
                    unsafe_allow_html=True
                )

            elif risk_level == "MEDIUM":

                st.markdown(
                    '<div class="risk-medium">'
                    f'MEDIUM RISK — {probability:.2f}%'
                    '</div>',
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    '<div class="risk-low">'
                    f'LOW RISK — {probability:.2f}%'
                    '</div>',
                    unsafe_allow_html=True
                )


            st.markdown("")


            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Risk Probability",
                    f"{probability:.2f}%"
                )

            with col2:

                st.metric(
                    "Risk Classification",
                    risk_level
                )

            with col3:

                st.metric(
                    "Model Prediction",
                    "Risk" if result["prediction"] == 1
                    else "No Risk"
                )


            st.markdown("---")


            st.subheader(
                "💡 Recommended Action"
            )

            if risk_level == "HIGH":

                st.error(
                    result["recommendation"]
                )

            elif risk_level == "MEDIUM":

                st.warning(
                    result["recommendation"]
                )

            else:

                st.success(
                    result["recommendation"]
                )


        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# BATCH ANALYSIS
# ============================================================

elif page == "Batch Analysis":

    st.title("📁 Batch Risk Analysis")

    st.markdown(
        """
        Upload a CSV containing DataCo-compatible order records.
        The system will calculate risk probabilities for every row.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )


    if uploaded_file is not None:

        try:

            df_upload = pd.read_csv(
                uploaded_file,
                encoding="latin1"
            )

            st.success(
                f"Loaded {len(df_upload):,} records."
            )

            st.dataframe(
                df_upload.head(10),
                use_container_width=True
            )


            if st.button(
                "Run Batch Prediction",
                type="primary"
            ):

                with st.spinner(
                    "Running risk predictions..."
                ):

                    X_upload = prepare_input(
                        df_upload,
                        model
                    )

                    probabilities = model.predict_proba(
                        X_upload
                    )[:, 1]

                    predictions = (
                        probabilities >= 0.50
                    ).astype(int)


                    results = df_upload.copy()

                    results["Risk Probability"] = (
                        probabilities * 100
                    ).round(2)

                    results["Risk Prediction"] = predictions

                    results["Risk Level"] = np.select(
                        [
                            probabilities >= 0.75,
                            probabilities >= 0.50
                        ],
                        [
                            "HIGH",
                            "MEDIUM"
                        ],
                        default="LOW"
                    )


                st.success(
                    "Batch prediction completed."
                )


                # --------------------------------------------
                # SUMMARY
                # --------------------------------------------

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.metric(
                        "Total Records",
                        f"{len(results):,}"
                    )

                with col2:

                    high_count = (
                        results["Risk Level"] == "HIGH"
                    ).sum()

                    st.metric(
                        "High Risk",
                        f"{high_count:,}"
                    )

                with col3:

                    avg_risk = (
                        results["Risk Probability"].mean()
                    )

                    st.metric(
                        "Average Risk",
                        f"{avg_risk:.2f}%"
                    )


                st.markdown("---")


                st.dataframe(
                    results,
                    use_container_width=True
                )


                # --------------------------------------------
                # DOWNLOAD
                # --------------------------------------------

                csv = results.to_csv(
                    index=False
                ).encode("utf-8")


                st.download_button(
                    "⬇️ Download Predictions",
                    data=csv,
                    file_name="supply_chain_risk_predictions.csv",
                    mime="text/csv"
                )


        except Exception as e:

            st.error(
                "Unable to process this CSV."
            )

            st.exception(e)


# ============================================================
# MODEL INTELLIGENCE
# ============================================================

elif page == "Model Intelligence":

    st.title("🧠 Model Intelligence")

    st.markdown(
        "Technical information about the trained prediction model."
    )

    st.markdown("---")


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    st.subheader("Verified Test Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", "93.24%")

    with col2:
        st.metric("Precision", "95.61%")

    with col3:
        st.metric("Recall", "91.90%")

    with col4:
        st.metric("F1 Score", "93.71%")

    with col5:
        st.metric("ROC-AUC", "98.70%")


    st.markdown("---")


    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader("Model Configuration")

    model_info = pd.DataFrame(
        {
            "Parameter": [
                "Algorithm",
                "Iterations",
                "Learning Rate",
                "Depth",
                "L2 Regularization",
                "Features",
                "Training Strategy"
            ],
            "Value": [
                "CatBoost Classifier",
                "5000",
                "0.03",
                "9",
                "7",
                "40",
                "70% Train / 15% Validation / 15% Test"
            ]
        }
    )

    st.table(model_info)


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    st.subheader("Feature Importance")

    try:

        importances = model.get_feature_importance()

        feature_names = model.feature_names_

        importance_df = pd.DataFrame(
            {
                "Feature": feature_names,
                "Importance": importances
            }
        ).sort_values(
            "Importance",
            ascending=False
        ).head(15)


        fig = px.bar(
            importance_df.sort_values(
                "Importance"
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top 15 Predictive Features"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    except Exception as e:

        st.warning(
            f"Feature importance unavailable: {e}"
        )


    # --------------------------------------------------------
    # ARCHITECTURE
    # --------------------------------------------------------

    st.subheader("System Architecture")

    st.code(
        """
DataCo Supply Chain Dataset
            │
            ▼
      Data Preprocessing
            │
            ▼
       Feature Engineering
            │
            ├── Temporal Features
            ├── Shipping Features
            ├── Geographic Features
            └── Interaction Features
            │
            ▼
       CatBoost Classifier
            │
            ▼
      Risk Probability
            │
            ▼
      Risk Classification
       ┌────┼────┐
       ▼    ▼    ▼
      LOW MEDIUM HIGH
       │    │    │
       └────┼────┘
            ▼
    Business Recommendation
        """,
        language="text"
    )


    st.info(
        """
        **Important:** The model predicts late-delivery risk
        based on patterns learned from the DataCo supply-chain
        dataset. It should be interpreted as a decision-support
        system rather than a guarantee of future shipment outcomes.
        """
    )