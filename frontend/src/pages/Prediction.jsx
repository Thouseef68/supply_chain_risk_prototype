import { useState } from "react";
import {
  AlertTriangle,
  CheckCircle2,
  Loader2,
  ShieldAlert,
  Truck,
} from "lucide-react";

import { predictRisk } from "../services/api";

import InfoBanner from "../components/InfoBanner";


const initialForm = {
  Type: "DEBIT",
  "Days for shipment (scheduled)": 4,
  "Benefit per order": 50,
  "Sales per customer": 100,
  "Category Name": "Sporting Goods",

  "Customer City": "New York City",
  "Customer Country": "Estados Unidos",
  "Customer Segment": "Consumer",
  "Customer State": "NY",

  "Department Name": "Outdoors",

  Latitude: 40.7128,
  Longitude: -74.0060,

  Market: "USCA",

  "Order City": "New York City",
  "Order Country": "Estados Unidos",

  "Order Item Discount": 0,
  "Order Item Discount Rate": 0,

  "Order Item Product Price": 100,
  "Order Item Profit Ratio": 0.2,
  "Order Item Quantity": 1,

  Sales: 100,

  "Order Item Total": 100,
  "Order Profit Per Order": 20,

  "Order Region": "East of USA",
  "Order State": "Nueva York",

  "Product Name": "Test Product",
  "Product Price": 100,

  "Shipping Mode": "Standard Class",

  "order date (DateOrders)": "2017-01-01",
};


function Input({
  label,
  value,
  onChange,
  type = "text",
}) {
  return (
    <div className="form-field">

      <label>
        {label}
      </label>

      <input
        type={type}
        value={value}
        onChange={(event) =>
          onChange(event.target.value)
        }
      />

    </div>
  );
}


function Prediction() {

  const [form, setForm] =
    useState(initialForm);

  const [result, setResult] =
    useState(null);

  const [loading, setLoading] =
    useState(false);

  const [error, setError] =
    useState(null);


  const updateField = (
    field,
    value
  ) => {

    setForm((previous) => ({
      ...previous,
      [field]: value,
    }));

  };


  const analyzeRisk = async () => {

    setLoading(true);
    setError(null);
    setResult(null);

    try {

      const numericFields = [
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Latitude",
        "Longitude",
        "Order Item Discount",
        "Order Item Discount Rate",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Product Price",
      ];

      const payload = {
        ...form
      };

      numericFields.forEach(
        (field) => {
          payload[field] =
            Number(payload[field]);
        }
      );

      const response =
        await predictRisk(payload);

      if (!response.success) {
        throw new Error(
          response.error ||
          "Prediction failed."
        );
      }

      setResult(
        response.result
      );

    } catch (err) {

      setError(
        err.message ||
        "Unable to connect to AI engine."
      );

    } finally {

      setLoading(false);

    }
  };


  return (
    <div className="prediction-page">

      <div className="page-header">

        <div>

          <div className="eyebrow">
            AI PREDICTION ENGINE
          </div>

          <h1>
            Shipment Risk Prediction
          </h1>

          <p>
            Analyze a shipment using the trained
            supply-chain risk model.
          </p>

        </div>

        <div className="header-status">
          <ShieldAlert size={18} />
          CatBoost AI
        </div>

      </div>


      <InfoBanner>
        <strong>
          How to use:
        </strong>{" "}
        Enter the shipment information and click{" "}
        <strong>
          Predict Risk
        </strong>
        . SupplyChainIQ processes the information
        through the trained model and returns the
        estimated risk probability and risk level.
      </InfoBanner>


      <div className="prediction-layout">

        {/* FORM */}

        <div className="panel">

          <div className="panel-header">

            <div>

              <h2>
                Shipment Information
              </h2>

              <p>
                Enter the available shipment characteristics
              </p>

            </div>

          </div>


          <div className="form-section">

            <div className="form-section-title">
              Shipping
            </div>

            <div className="form-grid">

              <div className="form-field">

                <label>
                  Shipping Mode
                </label>

                <select
                  value={
                    form["Shipping Mode"]
                  }
                  onChange={(e) =>
                    updateField(
                      "Shipping Mode",
                      e.target.value
                    )
                  }
                >

                  <option>
                    Standard Class
                  </option>

                  <option>
                    Second Class
                  </option>

                  <option>
                    First Class
                  </option>

                  <option>
                    Same Day
                  </option>

                </select>

              </div>


              <div className="form-field">

                <label>
                  Scheduled Shipping Days
                </label>

                <select
                  value={
                    form[
                      "Days for shipment (scheduled)"
                    ]
                  }
                  onChange={(e) =>
                    updateField(
                      "Days for shipment (scheduled)",
                      e.target.value
                    )
                  }
                >

                  <option value="0">
                    0
                  </option>

                  <option value="1">
                    1
                  </option>

                  <option value="2">
                    2
                  </option>

                  <option value="4">
                    4
                  </option>

                </select>

              </div>


              <div className="form-field">

                <label>
                  Market
                </label>

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
                  <option>Europe</option>
                  <option>LATAM</option>
                  <option>Pacific Asia</option>
                  <option>Africa</option>

                </select>

              </div>

            </div>

          </div>


          <div className="form-section">

            <div className="form-section-title">
              Customer
            </div>

            <div className="form-grid">

              <Input
                label="Customer City"
                value={
                  form["Customer City"]
                }
                onChange={(value) =>
                  updateField(
                    "Customer City",
                    value
                  )
                }
              />

              <Input
                label="Customer State"
                value={
                  form["Customer State"]
                }
                onChange={(value) =>
                  updateField(
                    "Customer State",
                    value
                  )
                }
              />

              <div className="form-field">

                <label>
                  Customer Segment
                </label>

                <select
                  value={
                    form[
                      "Customer Segment"
                    ]
                  }
                  onChange={(e) =>
                    updateField(
                      "Customer Segment",
                      e.target.value
                    )
                  }
                >

                  <option>
                    Consumer
                  </option>

                  <option>
                    Corporate
                  </option>

                  <option>
                    Home Office
                  </option>

                </select>

              </div>

            </div>

          </div>


          <div className="form-section">

            <div className="form-section-title">
              Order
            </div>

            <div className="form-grid">

              <Input
                label="Sales"
                type="number"
                value={form.Sales}
                onChange={(value) =>
                  updateField(
                    "Sales",
                    value
                  )
                }
              />

              <Input
                label="Product Price"
                type="number"
                value={
                  form["Product Price"]
                }
                onChange={(value) =>
                  updateField(
                    "Product Price",
                    value
                  )
                }
              />

              <Input
                label="Quantity"
                type="number"
                value={
                  form[
                    "Order Item Quantity"
                  ]
                }
                onChange={(value) =>
                  updateField(
                    "Order Item Quantity",
                    value
                  )
                }
              />

            </div>

          </div>


          <div className="form-section">

            <div className="form-section-title">
              Location & Date
            </div>

            <div className="form-grid">

              <Input
                label="Order City"
                value={
                  form["Order City"]
                }
                onChange={(value) =>
                  updateField(
                    "Order City",
                    value
                  )
                }
              />

              <Input
                label="Order Region"
                value={
                  form["Order Region"]
                }
                onChange={(value) =>
                  updateField(
                    "Order Region",
                    value
                  )
                }
              />

              <Input
                label="Order Date"
                type="date"
                value={
                  form[
                    "order date (DateOrders)"
                  ]
                }
                onChange={(value) =>
                  updateField(
                    "order date (DateOrders)",
                    value
                  )
                }
              />

            </div>

          </div>


          <button
            className="predict-button"
            onClick={analyzeRisk}
            disabled={loading}
          >

            {loading ? (
              <>
                <Loader2
                  size={18}
                  className="spin"
                />

                Analyzing shipment...
              </>
            ) : (
              <>
                <ShieldAlert size={18} />

                Analyze Risk
              </>
            )}

          </button>


          {error && (

            <div className="error-box">
              <AlertTriangle size={18} />
              {error}
            </div>

          )}

        </div>


        {/* RESULT */}

        <div className="result-column">

          {!result && !loading && (

            <div className="panel empty-result">

              <Truck size={38} />

              <h2>
                Awaiting Analysis
              </h2>

              <p>
                Enter shipment information and
                run the AI risk analysis.
              </p>

            </div>

          )}


          {loading && (

            <div className="panel empty-result">

              <Loader2
                size={38}
                className="spin"
              />

              <h2>
                Analyzing Shipment
              </h2>

              <p>
                The CatBoost model is evaluating
                the shipment characteristics.
              </p>

            </div>

          )}


          {result && (

            <>

              <div
                className={`result-card ${result.risk_level.toLowerCase()}`}
              >

                <div className="result-icon">

                  {result.risk_level === "HIGH" ? (
                    <AlertTriangle />
                  ) : result.risk_level === "MEDIUM" ? (
                    <ShieldAlert />
                  ) : (
                    <CheckCircle2 />
                  )}

                </div>

                <div className="result-label">
                  PREDICTED RISK
                </div>

                <div className="result-level">
                  {result.risk_level}
                </div>

                <div className="result-percentage">
                  {result.risk_percentage}%
                </div>

                <div className="result-description">
                  Late-delivery risk probability
                </div>

              </div>


              <div className="panel">

                <div className="panel-header">

                  <div>

                    <h2>
                      Recommended Action
                    </h2>

                    <p>
                      Operational response
                    </p>

                  </div>

                </div>


                <div className="recommendation">

                  <div className="recommendation-priority">
                    {result.recommendation.priority}
                  </div>

                  <p>
                    {result.recommendation.message}
                  </p>

                  <div className="action-list">

                    {result.recommendation.actions.map(
                      (action, index) => (

                        <div
                          className="action-item"
                          key={index}
                        >

                          <CheckCircle2 size={16} />

                          <span>
                            {action}
                          </span>

                        </div>

                      )
                    )}

                  </div>

                </div>

              </div>

            </>

          )}

        </div>

      </div>

    </div>
  );
}


export default Prediction;