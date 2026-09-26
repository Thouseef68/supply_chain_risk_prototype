import { useState } from "react";
import {
  SlidersHorizontal,
  AlertTriangle,
  Truck,
  RotateCcw,
  Play,
  Activity
} from "lucide-react";

import { whatIfPrediction } from "../services/api";

import InfoBanner from "../components/InfoBanner";


const disruptionDefaults = {
  Origin_Port: "Port of Shanghai",
  Destination_Port: "Port of Los Angeles",
  Transport_Mode: "Ocean",
  Product_Category: "Electronics",
  Distance_km: 8500,
  Weight_MT: 12,
  Fuel_Price_Index: 1.15,
  Geopolitical_Risk_Score: 45,
  Weather_Condition: "Normal",
  Carrier_Reliability_Score: 82,
  Lead_Time_Days: 18,
  Date: new Date().toISOString().split("T")[0]
};


const dataCoDefaults = {
  "Type": "DEBIT",
  "Days for shipment (scheduled)": 4,
  "Benefit per order": 50,
  "Sales per customer": 500,
  "Category Name": "Computers",
  "Customer City": "Los Angeles",
  "Customer Country": "United States",
  "Customer Segment": "Consumer",
  "Customer State": "CA",
  "Department Name": "Technology",
  "Latitude": 34.05,
  "Longitude": -118.24,
  "Market": "USCA",
  "Order City": "Los Angeles",
  "Order Country": "United States",
  "Order Item Discount": 10,
  "Order Item Discount Rate": 0.1,
  "Order Item Product Price": 100,
  "Order Item Profit Ratio": 0.2,
  "Order Item Quantity": 1,
  "Sales": 100,
  "Order Item Total": 90,
  "Order Profit Per Order": 20,
  "Order Region": "West",
  "Order State": "CA",
  "Product Name": "Product",
  "Product Price": 100,
  "Shipping Mode": "Standard Class",
  "order_year": new Date().getFullYear(),
  "order_month": new Date().getMonth() + 1,
  "order_day": new Date().getDate(),
  "order_dayofweek": new Date().getDay(),
  "order_week": 1,
  "ShippingMode_ScheduledDays": "Standard Class_4",
  "ShippingMode_Market": "Standard Class_USCA",
  "ShippingMode_Region": "Standard Class_West",
  "ShippingMode_Month": "Standard Class_1",
  "Market_ScheduledDays": "USCA_4",
  "Market_Region": "USCA_West",
  "Region_ScheduledDays": "West_4"
};


function WhatIf() {

  const [modelType, setModelType] = useState("disruption");

  const [form, setForm] = useState(disruptionDefaults);

  const [result, setResult] = useState(null);

  const [loading, setLoading] = useState(false);

  const [error, setError] = useState("");


  const changeModel = (type) => {

    setModelType(type);

    setResult(null);
    setError("");

    if (type === "disruption") {
      setForm(disruptionDefaults);
    } else {
      setForm(dataCoDefaults);
    }
  };


  const updateField = (key, value) => {

    setForm((previous) => ({
      ...previous,
      [key]: value
    }));

  };


  const resetForm = () => {

    setResult(null);
    setError("");

    setForm(
      modelType === "disruption"
        ? disruptionDefaults
        : dataCoDefaults
    );

  };


  const evaluateScenario = async () => {

    try {

      setLoading(true);
      setError("");
      setResult(null);

      const response = await whatIfPrediction(form);

      if (!response.success) {
        setError(
          response.error ||
          "Unable to evaluate scenario."
        );
        return;
      }

      setResult(response);

    } catch (err) {

      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to connect to prediction engine."
      );

    } finally {

      setLoading(false);

    }

  };


  const getRiskClass = (level) => {

    if (level === "HIGH") {
      return "whatif-risk high";
    }

    if (level === "MEDIUM") {
      return "whatif-risk medium";
    }

    return "whatif-risk low";
  };


  return (
    <div className="page-container">

      <div className="page-header">

        <div>

          <div className="eyebrow">
            SCENARIO SIMULATION
          </div>

          <h1>
            What-If Analysis
          </h1>

          <p>
            Modify operational conditions and evaluate
            how the prediction model responds.
          </p>

        </div>

      </div>


      <InfoBanner>
        <strong>
          How to use:
        </strong>{" "}
        Select the risk type, change the scenario
        variables, and click{" "}
        <strong>
          Evaluate Scenario
        </strong>
        . The result shows how the selected trained
        model responds to the supplied scenario.
      </InfoBanner>


      {/* MODEL SELECTOR */}

      <div className="whatif-model-selector">

        <button
          className={
            modelType === "disruption"
              ? "whatif-model active"
              : "whatif-model"
          }
          onClick={() => changeModel("disruption")}
        >

          <AlertTriangle size={21} />

          <div>

            <strong>
              Supply Chain Disruption
            </strong>

            <span>
              Disruption CatBoost
            </span>

          </div>

        </button>


        <button
          className={
            modelType === "dataco"
              ? "whatif-model active"
              : "whatif-model"
          }
          onClick={() => changeModel("dataco")}
        >

          <Truck size={21} />

          <div>

            <strong>
              Late Delivery
            </strong>

            <span>
              DataCo CatBoost
            </span>

          </div>

        </button>

      </div>


      <div className="whatif-layout">

        <div className="whatif-card">

          <div className="whatif-card-header">

            <div>

              <h2>
                Scenario Parameters
              </h2>

              <p>
                Adjust the variables you want to test.
              </p>

            </div>

            <SlidersHorizontal size={21} />

          </div>


          {modelType === "disruption" ? (

            <div className="whatif-form-grid">

              <div className="field">
                <label>Origin Port</label>

                <input
                  value={form.Origin_Port}
                  onChange={(e) =>
                    updateField(
                      "Origin_Port",
                      e.target.value
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Destination Port</label>

                <input
                  value={form.Destination_Port}
                  onChange={(e) =>
                    updateField(
                      "Destination_Port",
                      e.target.value
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Transport Mode</label>

                <select
                  value={form.Transport_Mode}
                  onChange={(e) =>
                    updateField(
                      "Transport_Mode",
                      e.target.value
                    )
                  }
                >
                  <option>Ocean</option>
                  <option>Air</option>
                  <option>Road</option>
                  <option>Rail</option>
                </select>

              </div>


              <div className="field">
                <label>Product Category</label>

                <input
                  value={form.Product_Category}
                  onChange={(e) =>
                    updateField(
                      "Product_Category",
                      e.target.value
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Distance (km)</label>

                <input
                  type="number"
                  value={form.Distance_km}
                  onChange={(e) =>
                    updateField(
                      "Distance_km",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Weight (MT)</label>

                <input
                  type="number"
                  value={form.Weight_MT}
                  onChange={(e) =>
                    updateField(
                      "Weight_MT",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Fuel Price Index</label>

                <input
                  type="number"
                  step="0.01"
                  value={form.Fuel_Price_Index}
                  onChange={(e) =>
                    updateField(
                      "Fuel_Price_Index",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Geopolitical Risk</label>

                <input
                  type="number"
                  min="0"
                  max="100"
                  value={form.Geopolitical_Risk_Score}
                  onChange={(e) =>
                    updateField(
                      "Geopolitical_Risk_Score",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Weather</label>

                <select
                  value={form.Weather_Condition}
                  onChange={(e) =>
                    updateField(
                      "Weather_Condition",
                      e.target.value
                    )
                  }
                >
                  <option>Normal</option>
                  <option>Rain</option>
                  <option>Storm</option>
                  <option>Snow</option>
                  <option>Extreme</option>
                </select>

              </div>


              <div className="field">
                <label>Carrier Reliability</label>

                <input
                  type="number"
                  min="0"
                  max="100"
                  value={form.Carrier_Reliability_Score}
                  onChange={(e) =>
                    updateField(
                      "Carrier_Reliability_Score",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Lead Time (days)</label>

                <input
                  type="number"
                  value={form.Lead_Time_Days}
                  onChange={(e) =>
                    updateField(
                      "Lead_Time_Days",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Date</label>

                <input
                  type="date"
                  value={form.Date}
                  onChange={(e) =>
                    updateField(
                      "Date",
                      e.target.value
                    )
                  }
                />
              </div>

            </div>

          ) : (

            <div className="whatif-form-grid">

              <div className="field">
                <label>Shipping Mode</label>

                <select
                  value={form["Shipping Mode"]}
                  onChange={(e) =>
                    updateField(
                      "Shipping Mode",
                      e.target.value
                    )
                  }
                >
                  <option>Standard Class</option>
                  <option>Second Class</option>
                  <option>First Class</option>
                  <option>Same Day</option>
                </select>

              </div>


              <div className="field">
                <label>Scheduled Shipping Days</label>

                <input
                  type="number"
                  value={form["Days for shipment (scheduled)"]}
                  onChange={(e) =>
                    updateField(
                      "Days for shipment (scheduled)",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Market</label>

                <select
                  value={form.Market}
                  onChange={(e) =>
                    updateField(
                      "Market",
                      e.target.value
                    )
                  }
                >
                  <option>USCA</option>
                  <option>LATAM</option>
                  <option>Europe</option>
                  <option>Pacific Asia</option>
                  <option>Africa</option>
                </select>

              </div>


              <div className="field">
                <label>Customer Segment</label>

                <select
                  value={form["Customer Segment"]}
                  onChange={(e) =>
                    updateField(
                      "Customer Segment",
                      e.target.value
                    )
                  }
                >
                  <option>Consumer</option>
                  <option>Corporate</option>
                  <option>Home Office</option>
                </select>

              </div>


              <div className="field">
                <label>Order Region</label>

                <input
                  value={form["Order Region"]}
                  onChange={(e) =>
                    updateField(
                      "Order Region",
                      e.target.value
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Order City</label>

                <input
                  value={form["Order City"]}
                  onChange={(e) =>
                    updateField(
                      "Order City",
                      e.target.value
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Product Price</label>

                <input
                  type="number"
                  value={form["Product Price"]}
                  onChange={(e) =>
                    updateField(
                      "Product Price",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Order Quantity</label>

                <input
                  type="number"
                  value={form["Order Item Quantity"]}
                  onChange={(e) =>
                    updateField(
                      "Order Item Quantity",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Sales</label>

                <input
                  type="number"
                  value={form.Sales}
                  onChange={(e) =>
                    updateField(
                      "Sales",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Order Item Discount</label>

                <input
                  type="number"
                  value={form["Order Item Discount"]}
                  onChange={(e) =>
                    updateField(
                      "Order Item Discount",
                      Number(e.target.value)
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Customer Country</label>

                <input
                  value={form["Customer Country"]}
                  onChange={(e) =>
                    updateField(
                      "Customer Country",
                      e.target.value
                    )
                  }
                />
              </div>


              <div className="field">
                <label>Order Country</label>

                <input
                  value={form["Order Country"]}
                  onChange={(e) =>
                    updateField(
                      "Order Country",
                      e.target.value
                    )
                  }
                />
              </div>

            </div>

          )}


          <div className="whatif-actions">

            <button
              className="secondary-btn"
              onClick={resetForm}
            >
              <RotateCcw size={16} />
              Reset
            </button>


            <button
              className="primary-btn"
              onClick={evaluateScenario}
              disabled={loading}
            >

              <Play size={16} />

              {loading
                ? "Evaluating..."
                : "Evaluate Scenario"}

            </button>

          </div>

        </div>


        {/* RESULT */}

        <div className="whatif-result-card">

          {!result && !loading && (

            <div className="whatif-result-empty">

              <Activity size={42} />

              <h3>
                Scenario Result
              </h3>

              <p>
                Modify the parameters and click
                Evaluate Scenario.
              </p>

            </div>

          )}


          {loading && (

            <div className="whatif-result-empty">

              <Activity size={42} />

              <h3>
                Evaluating scenario...
              </h3>

              <p>
                Running the selected CatBoost model.
              </p>

            </div>

          )}


          {error && (

            <div className="whatif-error">

              <AlertTriangle size={20} />

              <span>
                {error}
              </span>

            </div>

          )}


          {result && (

            <div className="whatif-result">

              <div className="result-label">
                PREDICTED RISK
              </div>

              <div
                className={getRiskClass(
                  result.result.risk_level
                )}
              >
                {result.result.risk_level}
              </div>


              <div className="result-probability">

                <span>
                  Risk Probability
                </span>

                <strong>
                  {result.result.risk_percentage}%
                </strong>

              </div>


              <div className="result-details">

                <div>
                  <span>Model</span>
                  <strong>
                    {result.model}
                  </strong>
                </div>

                <div>
                  <span>Prediction Type</span>
                  <strong>
                    {result.risk_type}
                  </strong>
                </div>

              </div>


              <div className="scenario-note">

                <strong>
                  Scenario assessment
                </strong>

                <p>
                  This result represents the selected
                  model's response to the supplied scenario.
                  It should not be interpreted as a causal
                  effect of changing an individual variable.
                </p>

              </div>

            </div>

          )}

        </div>

      </div>

    </div>
  );
}

export default WhatIf;