# SupplyChainIQ

<p align="center">
  <strong>Predict Risk. Understand Causes. Act Before Disruption.</strong>
</p>

<p align="center">
  Machine-learning powered supply-chain risk intelligence built with React, FastAPI and CatBoost.
</p>

<p align="center">

![Status](https://img.shields.io/badge/status-functional%20prototype-2563eb)
![Frontend](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb)
![Backend](https://img.shields.io/badge/backend-FastAPI-009688)
![ML](https://img.shields.io/badge/ML-CatBoost-ffcc00)
![Database](https://img.shields.io/badge/database-SQLite-003b57)
![License](https://img.shields.io/badge/license-Educational-lightgrey)

</p>

---

## What is SupplyChainIQ?

**SupplyChainIQ** is a supply-chain risk intelligence prototype that turns shipment and operational data into understandable machine-learning predictions.

Instead of exposing only a raw ML model, the platform provides a complete workflow for:

- individual risk prediction
- bulk shipment analysis
- risk analytics
- shipment exploration
- What-If scenario analysis
- model intelligence
- prediction history
- decision-oriented visualization

The system currently supports **two separate prediction tasks**:

| Model | Purpose |
|---|---|
| **DataCo CatBoost** | Late Delivery Risk |
| **Disruption CatBoost** | Supply Chain Disruption Risk |

These models are intentionally kept separate because they predict different outcomes.

---

## Product Flow

```text
                         ┌──────────────────────┐
                         │      SupplyChainIQ   │
                         └──────────┬───────────┘
                                    │
                         Shipment / Operational Data
                                    │
                         ┌──────────▼───────────┐
                         │    FastAPI Backend    │
                         │   Hybrid Model Router │
                         └──────────┬───────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
          ┌────────▼────────┐              ┌─────────▼────────┐
          │ DataCo CatBoost │              │ Disruption       │
          │ Late Delivery   │              │ CatBoost         │
          │ Risk            │              │ Disruption Risk  │
          └────────┬────────┘              └─────────┬────────┘
                   │                                 │
                   └────────────────┬────────────────┘
                                    │
                            Risk Probability
                                    │
                           LOW / MEDIUM / HIGH
                                    │
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
          Analytics             What-If              History
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    │
                              Decision Support
```

---

## ✨ Platform Highlights

### Command Center
A centralized operational dashboard showing:

- model status
- shipment risk analytics
- prediction activity
- risk distribution
- recent assessment information

### Risk Prediction
Run an individual prediction from shipment information and receive:

- predicted class
- probability
- percentage
- risk level
- model used
- prediction type

### Batch Analysis
Upload supported CSV datasets and analyze up to **6,000 records per batch**.

The backend detects the dataset schema and routes it to the appropriate model.

### Risk Analytics
Explore risk patterns through interactive charts covering:

- regions
- shipment types
- monthly patterns
- risk distributions

### Shipment Explorer
Search and filter shipment records by:

- city
- state
- country
- market
- shipping mode
- risk
- customer
- product

### Model Intelligence
Inspect:

- model purpose
- evaluation metrics
- feature importance
- model routing
- prediction context

### What-If Analysis
Modify operational conditions and evaluate how a selected trained model responds.

Examples include:

- distance
- shipment weight
- fuel price index
- geopolitical risk
- weather
- carrier reliability
- lead time
- transport mode
- product category
- shipping mode
- market

### Prediction History
Prediction results are stored locally in SQLite so previous assessments remain available after refreshing the application.

### How It Works
A built-in guide explains the platform in human-readable language, including what it does, how the models work, how to use each section, and who can use it.

---

# 🧠 Machine Learning

## DataCo CatBoost

**Task:** Late Delivery Risk

The model uses a leakage-aware, feature-engineered representation of the DataCo supply-chain dataset with **40 final features**.

The feature set includes operational, customer, product, geographic, market, date-derived, and engineered interaction features.

### Evaluation

Held-out test set: **27,078 samples**

| Metric | Result |
|:--|--:|
| Accuracy | **93.24%** |
| Precision | **95.61%** |
| Recall | **91.90%** |
| F1 Score | **93.71%** |
| ROC-AUC | **98.70%** |

---

## Disruption CatBoost

**Task:** Supply Chain Disruption Risk

The model uses disruption-oriented operational information such as:

- origin port
- destination port
- transport mode
- product category
- distance
- weight
- fuel price index
- geopolitical risk
- weather
- carrier reliability
- lead time
- date-derived features

### Evaluation

Held-out test set: **1,000 samples**

| Metric | Result |
|:--|--:|
| Accuracy | **73.40%** |
| Precision | **78.58%** |
| Recall | **77.81%** |
| F1 Score | **78.20%** |
| ROC-AUC | **82.91%** |

> **Interpretation:** These metrics describe performance on the available evaluation datasets. They do not guarantee future real-world performance.

---

# 🏗️ Architecture

```text
Frontend
React + Vite
│
│ Axios / REST
▼
Backend
FastAPI
│
├── Prediction Engine
├── Hybrid Schema Detection
├── Analytics
├── Batch Processing
├── What-If Engine
└── Prediction History
│
├── CatBoost Models
│
└── SQLite
```

### Frontend

- React
- Vite
- Axios
- Recharts
- Lucide React
- Custom CSS

### Backend

- Python
- FastAPI
- Pandas
- NumPy
- Joblib
- SQLite

### Machine Learning

- CatBoost
- Classification
- Feature engineering
- Probability-based risk estimation

---

# 📁 Project Structure

```text
supply_chain_risk_prototype/
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── App.jsx
│   │   ├── main.jsx
│   │   └── index.css
│   ├── package.json
│   └── vite.config.js
│
├── backend/
│   ├── main.py
│   ├── predict.py
│   ├── hybrid_predict.py
│   ├── database.py
│   └── requirements.txt
│
├── data/
│   └── dataco/
│       └── DataCoSupplyChainDataset.csv
│
├── models/
│   ├── dataco_catboost_features_model.joblib
│   ├── disruption_catboost_model.joblib
│   └── disruption_model_metadata.json
│
├── outputs/
│   └── supplychainiq.db
│
├── src/
│
├── .gitignore
└── README.md
```

---

# 🚀 Run Locally

## 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd supply_chain_risk_prototype
```

---

## 2. Start the FastAPI backend

### Windows

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 3. Start the React frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

# 🔌 API

| Method | Endpoint | Purpose |
|:--|:--|:--|
| GET | `/api/health` | Backend health |
| GET | `/api/model` | Model information |
| GET | `/api/dashboard` | Dashboard data |
| GET | `/api/analytics` | Risk analytics |
| POST | `/api/predict` | Individual prediction |
| POST | `/api/batch-predict` | Bulk prediction |
| GET | `/api/batch-template` | Batch template |
| GET | `/api/shipments` | Shipment explorer |
| GET | `/api/shipment-filters` | Explorer filters |
| GET | `/api/model-intelligence` | Model metrics and intelligence |
| POST | `/api/what-if` | Scenario prediction |
| GET | `/api/prediction-history` | Stored predictions |
| DELETE | `/api/prediction-history` | Clear prediction history |
| GET | `/api/history-summary` | Prediction summary |

---

# 🔄 Prediction Logic

## Individual Prediction

```text
User Input
   ↓
FastAPI
   ↓
Feature Engineering
   ↓
CatBoost
   ↓
Probability
   ↓
Risk Level
   ↓
SQLite History
```

## Batch Prediction

```text
CSV Upload
   ↓
Schema Detection
   ↓
┌───────────────┬─────────────────────┐
│ DataCo schema │ Disruption schema   │
└───────┬───────┴──────────┬──────────┘
        ↓                  ↓
 DataCo CatBoost     Disruption CatBoost
        ↓                  ↓
        └─────────┬────────┘
                  ↓
             Batch Results
```

---

# 🎯 Risk Interpretation

For the disruption workflow:

```text
≥ 70%  → HIGH
50–69.99% → MEDIUM
< 50%  → LOW
```

The displayed risk level is an application interpretation of model probability.

It should be considered alongside the prediction context rather than treated as a guaranteed outcome.

---

# 👥 Intended Users

SupplyChainIQ is designed as a prototype decision-support system for:

- Supply-chain managers
- Logistics teams
- Operations teams
- Procurement teams
- Data analysts
- Researchers
- Students

Potential uses include:

- identifying shipments requiring additional attention
- screening large shipment datasets
- exploring historical risk patterns
- comparing operational scenarios
- demonstrating machine-learning based supply-chain analytics

---

# ⚠️ Important Model Notes

SupplyChainIQ is a **predictive decision-support prototype**, not a production guarantee system.

### Probability ≠ certainty

A prediction such as `82%` means the model estimates a high probability based on the supplied features. It does not mean the event is guaranteed.

### Feature importance ≠ causation

Feature importance shows how strongly a feature contributes to model predictions relative to other features. It does not prove that the feature causes the outcome.

### What-If ≠ causal experiment

What-If Analysis shows how the trained model responds to a changed input scenario. It does not prove that changing one variable alone will cause the predicted change in the real world.

### Two models, two tasks

The DataCo model predicts **late-delivery risk**.

The Disruption model predicts **supply-chain disruption risk**.

They are separate models trained for separate prediction tasks.

---

# 🔐 Data & Privacy

Prediction history is stored locally in:

```text
outputs/supplychainiq.db
```

The prototype does not require a separate database server.

Before publishing the project publicly, review dataset licensing and remove any sensitive or restricted data.

---

# 📈 Performance Considerations

The prototype includes:

- cached loading of the large DataCo dataset
- local SQLite persistence
- model loading through the backend
- schema-based model routing
- a 6,000-record batch limit

Potential production improvements:

- Parquet or indexed database storage
- vectorized batch inference
- background processing
- cloud database
- authentication
- monitoring
- model drift detection
- automated retraining

---

# 🛣️ Future Roadmap

Possible future improvements:

- [ ] SHAP-based individual explanations
- [ ] Real-time weather integration
- [ ] Fuel-price feeds
- [ ] Geopolitical event feeds
- [ ] Carrier performance monitoring
- [ ] Automated alerts
- [ ] Authentication and role-based access
- [ ] Cloud database
- [ ] Model drift monitoring
- [ ] Automated model retraining
- [ ] Production monitoring

---

# 📚 Data & Research Context

The project uses:

- **DataCo SMART SUPPLY CHAIN FOR BIG DATA ANALYSIS** for the late-delivery workflow.
- A separate supply-chain disruption dataset for the disruption workflow.

The DataCo workflow uses leakage-aware preprocessing, including removal of fields that directly expose delivery outcomes or information unavailable at prediction time.

---

# 🧪 Project Status

**Functional Prototype**

Current capabilities:

- ✅ React web application
- ✅ FastAPI REST API
- ✅ DataCo CatBoost model
- ✅ Disruption CatBoost model
- ✅ Hybrid model routing
- ✅ Individual prediction
- ✅ Batch prediction
- ✅ Risk analytics
- ✅ Shipment Explorer
- ✅ Model Intelligence
- ✅ What-If Analysis
- ✅ Prediction History
- ✅ SQLite persistence
- ✅ Human-readable system guide
- ✅ Enterprise-style interface

---

# 📄 License

This project is intended for educational, research, and prototype purposes.

Third-party datasets and assets remain subject to their respective licenses and usage terms.

---

<p align="center">
  <strong>SupplyChainIQ</strong><br>
  Predict Risk. Understand Causes. Act Before Disruption.
</p>
