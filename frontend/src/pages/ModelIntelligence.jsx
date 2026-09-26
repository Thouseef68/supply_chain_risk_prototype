import { useEffect, useState } from "react";

import {
  BrainCircuit,
  Target,
  Activity,
  Layers3,
  BarChart3,
  ShieldCheck,
  Database,
} from "lucide-react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from "recharts";

import {
  getModelIntelligence,
} from "../services/api";

import InfoBanner from "../components/InfoBanner";


export default function ModelIntelligence() {

  const [data, setData] = useState(null);

  const [loading, setLoading] =
    useState(true);

  const [error, setError] =
    useState("");

  useEffect(() => {
    loadModel();
  }, []);


  const loadModel = async () => {

    try {

      setLoading(true);

      const result =
        await getModelIntelligence();

      if (!result.success) {
        throw new Error(
          result.error ||
          "Unable to load model information."
        );
      }

      setData(result);

    } catch (err) {

      setError(
        err.message ||
        "Unable to load model information."
      );

    } finally {

      setLoading(false);

    }
  };


  if (loading) {

    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>

        <p>
          Loading model intelligence...
        </p>
      </div>
    );
  }


  if (error) {

    return (
      <div className="analytics-error">

        <BrainCircuit size={22} />

        <div>

          <h3>
            Model information unavailable
          </h3>

          <p>
            {error}
          </p>

        </div>

      </div>
    );
  }


  const dataco =
    data.models.dataco;

  const disruption =
    data.models.disruption;


  return (

    <div className="model-page">

      {/* HEADER */}

      <div className="page-header">

        <div>

          <div className="page-kicker">

            <BrainCircuit size={16} />

            MODEL INTELLIGENCE

          </div>

          <h1>
            Model Intelligence
          </h1>

          <p>
            Inspect the prediction models,
            evaluation metrics and feature
            importance used by SupplyChainIQ.
          </p>

        </div>

      </div>


      <InfoBanner>
        <strong>
          What you're seeing:
        </strong>{" "}
        This page explains the models used by
        SupplyChainIQ, their evaluation results, and
        the features that have the greatest influence
        on model predictions.
      </InfoBanner>


      {/* MODEL CARDS */}

      <div className="model-overview-grid">

        <div className="model-overview-card">

          <div className="model-card-top">

            <div className="model-icon blue">
              <Target size={22} />
            </div>

            <span className="model-status">
              ACTIVE
            </span>

          </div>

          <h2>
            {dataco.name}
          </h2>

          <p>
            {dataco.purpose}
          </p>

          <div className="model-card-meta">

            <span>
              <Database size={14} />
              40 features
            </span>

            <span>
              <Layers3 size={14} />
              CatBoost
            </span>

          </div>

        </div>


        <div className="model-overview-card">

          <div className="model-card-top">

            <div className="model-icon purple">
              <Activity size={22} />
            </div>

            <span className="model-status">
              ACTIVE
            </span>

          </div>

          <h2>
            {disruption.name}
          </h2>

          <p>
            {disruption.purpose}
          </p>

          <div className="model-card-meta">

            <span>
              <Database size={14} />
              17 features
            </span>

            <span>
              <Layers3 size={14} />
              CatBoost
            </span>

          </div>

        </div>

      </div>


      {/* DATACO */}

      <ModelSection
        model={dataco}
        type="dataco"
      />


      {/* DISRUPTION */}

      <ModelSection
        model={disruption}
        type="disruption"
      />


      {/* ROUTING */}

      <div className="panel">

        <div className="panel-header">

          <div>

            <h2>
              Automatic Model Routing
            </h2>

            <span>
              SupplyChainIQ selects the model
              according to the uploaded schema.
            </span>

          </div>

          <BrainCircuit size={20} />

        </div>


        <div className="routing-flow">

          <div className="routing-node">

            <Database size={21} />

            <strong>
              CSV Upload
            </strong>

            <span>
              Shipment dataset
            </span>

          </div>


          <div className="routing-arrow">
            →
          </div>


          <div className="routing-node">

            <BrainCircuit size={21} />

            <strong>
              Schema Detection
            </strong>

            <span>
              Automatic classification
            </span>

          </div>


          <div className="routing-arrow">
            →
          </div>


          <div className="routing-split">

            <div className="routing-model">

              <ShieldCheck size={18} />

              <strong>
                DataCo CatBoost
              </strong>

              <span>
                Late Delivery
              </span>

            </div>


            <div className="routing-model purple-routing">

              <ShieldCheck size={18} />

              <strong>
                Disruption CatBoost
              </strong>

              <span>
                Supply Chain Disruption
              </span>

            </div>

          </div>

        </div>

      </div>

    </div>
  );
}


/* =================================
   MODEL SECTION
================================= */

function ModelSection({
  model,
  type,
}) {

  const metrics = [
    {
      label: "Accuracy",
      value: metricsPercent(
        model.metrics.accuracy
      ),
      icon: Target,
    },
    {
      label: "Precision",
      value: metricsPercent(
        model.metrics.precision
      ),
      icon: ShieldCheck,
    },
    {
      label: "Recall",
      value: metricsPercent(
        model.metrics.recall
      ),
      icon: Activity,
    },
    {
      label: "F1 Score",
      value: metricsPercent(
        model.metrics.f1
      ),
      icon: BarChart3,
    },
    {
      label: "ROC-AUC",
      value: metricsPercent(
        model.metrics.roc_auc
      ),
      icon: BrainCircuit,
    },
  ];


  return (

    <div className="panel model-section">

      <div className="panel-header">

        <div>

          <h2>
            {model.name}
          </h2>

          <span>
            {model.purpose}
          </span>

        </div>

        <div className="model-test-info">

          Test samples:
          {" "}
          {model.metrics.test_samples.toLocaleString()}

        </div>

      </div>


      {/* METRICS */}

      <div className="metric-grid">

        {metrics.map(
          (metric) => {

            const Icon =
              metric.icon;

            return (

              <div
                className="model-metric"
                key={metric.label}
              >

                <div className="model-metric-icon">
                  <Icon size={17} />
                </div>

                <span>
                  {metric.label}
                </span>

                <strong>
                  {metric.value}%
                </strong>

              </div>

            );
          }
        )}

      </div>


      {/* EVALUATION NOTE */}

      <p className="model-eval-note">
        <strong>
          Note:
        </strong>{" "}
        Evaluation metrics are based on held-out test
        data used during model development. They
        indicate model performance on those evaluation
        datasets and do not guarantee future real-world
        outcomes.
      </p>


      {/* FEATURES */}

      <div className="feature-section">

        <div className="feature-section-header">

          <div>

            <h3>
              Feature Importance
            </h3>

            <p>
              Feature importance shows how strongly each
              feature contributes to the model's
              predictions relative to the other features.
              It should not be interpreted as proof that
              changing a feature will directly cause the
              predicted outcome.
            </p>

          </div>

        </div>


        <div className="feature-chart">

          <ResponsiveContainer
            width="100%"
            height={
              Math.max(
                300,
                model.features.length * 36
              )
            }
          >

            <BarChart
              data={model.features}
              layout="vertical"
              margin={{
                left: 20,
                right: 30,
                top: 10,
                bottom: 10,
              }}
            >

              <CartesianGrid
                strokeDasharray="3 3"
                opacity={0.1}
              />

              <XAxis
                type="number"
              />

              <YAxis
                type="category"
                dataKey="feature"
                width={180}
                tick={{
                  fill: "#858da0",
                  fontSize: 11,
                }}
              />

              <Tooltip
                contentStyle={{
                  backgroundColor: "#111827",
                  border:
                    "1px solid rgba(255,255,255,0.1)",
                  borderRadius: "9px",
                }}
                formatter={(value) => [
                  value,
                  "Importance",
                ]}
              />

              <Bar
                dataKey="importance"
                fill={
                  type === "dataco"
                    ? "#3b82f6"
                    : "#8b5cf6"
                }
                radius={[
                  0,
                  5,
                  5,
                  0,
                ]}
              />

            </BarChart>

          </ResponsiveContainer>

        </div>

      </div>

    </div>
  );
}


function metricsPercent(value) {

  return Number(value).toFixed(2);

}