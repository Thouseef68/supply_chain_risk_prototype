import { useEffect, useState } from "react";
import {
  History,
  RefreshCw,
  Trash2,
  ShieldAlert,
  ShieldCheck,
  Activity
} from "lucide-react";

import {
  getPredictionHistory,
  clearPredictionHistory
} from "../services/api";

import InfoBanner from "../components/InfoBanner";


function PredictionHistory() {

  const [records, setRecords] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadHistory = async () => {

    try {

      setLoading(true);
      setError("");

      const data = await getPredictionHistory(100);

      if (data.success) {
        setRecords(data.records || []);
      } else {
        setError(data.error || "Unable to load prediction history.");
      }

    } catch (err) {

      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to connect to backend."
      );

    } finally {

      setLoading(false);

    }
  };


  useEffect(() => {
    loadHistory();
  }, []);


  const handleClear = async () => {

    const confirmed = window.confirm(
      "Clear all prediction history?"
    );

    if (!confirmed) return;

    try {

      await clearPredictionHistory();

      setRecords([]);

    } catch (err) {

      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to clear history."
      );

    }
  };


  const getRiskClass = (level) => {

    if (level === "HIGH") return "history-risk high";

    if (level === "MEDIUM") return "history-risk medium";

    return "history-risk low";
  };


  return (
    <div className="page-container">

      <div className="page-header">

        <div>
          <div className="eyebrow">
            DECISION HISTORY
          </div>

          <h1>
            Prediction History
          </h1>

          <p>
            Review previous SupplyChainIQ risk assessments
            and model decisions.
          </p>
        </div>


        <div className="history-actions">

          <button
            className="secondary-btn"
            onClick={loadHistory}
          >
            <RefreshCw size={17} />
            Refresh
          </button>


          {records.length > 0 && (
            <button
              className="danger-btn"
              onClick={handleClear}
            >
              <Trash2 size={17} />
              Clear History
            </button>
          )}

        </div>

      </div>


      <InfoBanner>
        <strong>
          What you're seeing:
        </strong>{" "}
        Predictions generated through the application
        are stored locally so previous assessments
        can be reviewed later.
      </InfoBanner>


      {error && (
        <div className="error-banner">
          {error}
        </div>
      )}


      <div className="history-summary">

        <div className="history-stat">

          <div className="history-stat-icon">
            <History size={20} />
          </div>

          <div>
            <span>Total Predictions</span>
            <strong>{records.length}</strong>
          </div>

        </div>


        <div className="history-stat">

          <div className="history-stat-icon">
            <ShieldAlert size={20} />
          </div>

          <div>
            <span>High Risk</span>

            <strong>
              {
                records.filter(
                  item => item.risk_level === "HIGH"
                ).length
              }
            </strong>
          </div>

        </div>


        <div className="history-stat">

          <div className="history-stat-icon">
            <ShieldCheck size={20} />
          </div>

          <div>
            <span>Low Risk</span>

            <strong>
              {
                records.filter(
                  item => item.risk_level === "LOW"
                ).length
              }
            </strong>
          </div>

        </div>


        <div className="history-stat">

          <div className="history-stat-icon">
            <Activity size={20} />
          </div>

          <div>
            <span>Models Used</span>

            <strong>
              {
                new Set(
                  records.map(item => item.model)
                ).size
              }
            </strong>
          </div>

        </div>

      </div>


      <div className="history-card">

        {loading ? (

          <div className="history-empty">
            Loading prediction history...
          </div>

        ) : records.length === 0 ? (

          <div className="history-empty">

            <History size={42} />

            <h3>
              No predictions yet
            </h3>

            <p>
              Predictions made from Risk Prediction
              and What-If Analysis will appear here.
            </p>

          </div>

        ) : (

          <div className="history-table-wrapper">

            <table className="history-table">

              <thead>

                <tr>
                  <th>ID</th>
                  <th>Date</th>
                  <th>Model</th>
                  <th>Risk Type</th>
                  <th>Prediction</th>
                  <th>Probability</th>
                  <th>Risk</th>
                </tr>

              </thead>


              <tbody>

                {records.map((record) => (

                  <tr key={record.id}>

                    <td>
                      #{record.id}
                    </td>

                    <td>
                      {new Date(
                        record.created_at
                      ).toLocaleString()}
                    </td>

                    <td>
                      <span className="model-badge">
                        {record.model}
                      </span>
                    </td>

                    <td>
                      {record.risk_type}
                    </td>

                    <td>
                      {record.prediction === 1
                        ? "Risk Detected"
                        : "No Risk"}
                    </td>

                    <td>
                      {Number(
                        record.risk_percentage
                      ).toFixed(2)}%
                    </td>

                    <td>
                      <span
                        className={getRiskClass(
                          record.risk_level
                        )}
                      >
                        {record.risk_level}
                      </span>
                    </td>

                  </tr>

                ))}

              </tbody>

            </table>

          </div>

        )}

        </div>

      </div>

  );
}

export default PredictionHistory;