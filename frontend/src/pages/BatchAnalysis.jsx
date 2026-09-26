import { useState } from "react";
import {
  Upload,
  FileSpreadsheet,
  Download,
  ShieldAlert,
  ShieldCheck,
  AlertTriangle,
  RefreshCw,
} from "lucide-react";

import {
  batchPredict,
  getBatchTemplate,
} from "../services/api";

import InfoBanner from "../components/InfoBanner";

export default function BatchAnalysis() {

  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);

  const [loading, setLoading] = useState(false);
  const [templateLoading, setTemplateLoading] = useState(false);

  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const handleFile = (selectedFile) => {

    if (!selectedFile) return;

    if (
      !selectedFile.name
        .toLowerCase()
        .endsWith(".csv")
    ) {
      setError("Please select a CSV file.");
      return;
    }

    setFile(selectedFile);
    setResult(null);
    setError("");
  };

  const handleInputChange = (event) => {
    handleFile(event.target.files[0]);
  };

  const handleDrop = (event) => {

    event.preventDefault();

    setDragActive(false);

    const droppedFile =
      event.dataTransfer.files[0];

    handleFile(droppedFile);
  };

  const runAnalysis = async () => {

    if (!file) {
      setError("Please select a CSV file first.");
      return;
    }

    try {

      setLoading(true);
      setError("");
      setResult(null);

      const response =
        await batchPredict(file);

      if (!response.success) {
        throw new Error(
          response.error ||
          "Batch prediction failed."
        );
      }

      setResult(response);

    } catch (err) {

      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to process the CSV."
      );

    } finally {

      setLoading(false);

    }
  };

  const downloadTemplate = async () => {

    try {

      setTemplateLoading(true);
      setError("");

      const response =
        await getBatchTemplate();

      if (!response.success) {
        throw new Error(response.error);
      }

      const blob = new Blob(
        [response.csv],
        { type: "text/csv;charset=utf-8;" }
      );

      const url =
        URL.createObjectURL(blob);

      const link =
        document.createElement("a");

      link.href = url;
      link.download =
        response.filename ||
        "supply_chain_batch_template.csv";

      document.body.appendChild(link);
      link.click();

      document.body.removeChild(link);

      URL.revokeObjectURL(url);

    } catch (err) {

      setError(
        err.message ||
        "Unable to download template."
      );

    } finally {

      setTemplateLoading(false);

    }
  };

  const downloadResults = () => {

    if (!result?.data) return;

    const headers =
      Object.keys(result.data[0]);

    const escapeCSV = (value) => {

      if (
        value === null ||
        value === undefined
      ) {
        return "";
      }

      const text =
        String(value);

      if (
        text.includes(",") ||
        text.includes('"') ||
        text.includes("\n")
      ) {
        return `"${text.replace(
          /"/g,
          '""'
        )}"`;
      }

      return text;
    };

    const csv = [
      headers.join(","),
      ...result.data.map((row) =>
        headers
          .map((header) =>
            escapeCSV(row[header])
          )
          .join(",")
      ),
    ].join("\n");

    const blob = new Blob(
      [csv],
      { type: "text/csv;charset=utf-8;" }
    );

    const url =
      URL.createObjectURL(blob);

    const link =
      document.createElement("a");

    link.href = url;
    link.download =
      "supply_chain_risk_results.csv";

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);

    URL.revokeObjectURL(url);
  };

  return (
    <div className="batch-page">

      {/* HEADER */}

      <div className="page-header">

        <div>

          <div className="page-kicker">
            <FileSpreadsheet size={16} />
            OPERATIONAL RISK SCORING
          </div>

          <h1>Batch Analysis</h1>

          <p>
            Upload multiple shipment records and
            evaluate them using the trained CatBoost
            risk model.
          </p>

        </div>

      </div>

      <InfoBanner>
        <strong>
          How to use:
        </strong>{" "}
        Upload a supported CSV file. SupplyChainIQ
        identifies the dataset type, selects the
        appropriate model, and evaluates multiple
        shipments together.
      </InfoBanner>

      {/* UPLOAD */}

      <div className="panel batch-upload-panel">

        <div
          className={`batch-dropzone ${
            dragActive ? "drag-active" : ""
          } ${
            file ? "has-file" : ""
          }`}
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() =>
            setDragActive(false)
          }
          onDrop={handleDrop}
        >

          <input
            type="file"
            accept=".csv"
            id="batch-file"
            onChange={handleInputChange}
            hidden
          />

          <label
            htmlFor="batch-file"
            className="batch-upload-content"
          >

            <div className="upload-icon">

              {file ? (
                <FileSpreadsheet size={30} />
              ) : (
                <Upload size={30} />
              )}

            </div>

            {file ? (
              <>
                <h3>{file.name}</h3>

                <p>
                  {(file.size / 1024 / 1024)
                    .toFixed(2)} MB
                </p>
              </>
            ) : (
              <>
                <h3>
                  Drop your CSV file here
                </h3>

                <p>
                  or click to browse from your
                  computer
                </p>
              </>
            )}

          </label>

        </div>

        <div className="batch-actions">

          <button
            className="primary-button"
            onClick={runAnalysis}
            disabled={!file || loading}
          >

            {loading ? (
              <>
                <RefreshCw
                  size={17}
                  className="spin"
                />
                Processing...
              </>
            ) : (
              <>
                <ShieldAlert size={17} />
                Analyze Shipments
              </>
            )}

          </button>

          <button
            className="secondary-button"
            onClick={downloadTemplate}
            disabled={templateLoading}
          >

            <Download size={17} />

            {templateLoading
              ? "Preparing..."
              : "Download Sample CSV"}

          </button>

        </div>

        <div className="batch-info">

          <span>
            Maximum 6,000 shipments per batch
          </span>

          <span>
            CSV format only
          </span>

          <span>
            Powered by CatBoost
          </span>

        </div>

      </div>

      {/* ERROR */}

      {error && (

        <div className="batch-error">

          <AlertTriangle size={21} />

          <div>
            <strong>
              Processing Error
            </strong>

            <p>{error}</p>
          </div>

        </div>

      )}

      {/* RESULTS */}

      {result && (

        <>

          <div className="batch-results-header">

            <div>

              <div className="page-kicker">
                ANALYSIS COMPLETE
              </div>

              <h2>
                Batch Risk Results
              </h2>

              <p>
                {result.total_rows.toLocaleString()}
                {" "}shipments evaluated.
              </p>

            </div>

            <button
              className="primary-button"
              onClick={downloadResults}
            >
              <Download size={17} />
              Download Results
            </button>

          </div>

          {/* KPI CARDS */}

          <div className="batch-kpi-grid">

            <div className="batch-kpi">

              <div className="batch-kpi-icon total">
                <FileSpreadsheet size={21} />
              </div>

              <div>
                <span>Total Shipments</span>
                <strong>
                  {result.total_rows.toLocaleString()}
                </strong>
              </div>

            </div>

            <div className="batch-kpi high">

              <div className="batch-kpi-icon">
                <ShieldAlert size={21} />
              </div>

              <div>
                <span>High Risk</span>
                <strong>
                  {result.high_risk}
                </strong>
              </div>

            </div>

            <div className="batch-kpi medium">

              <div className="batch-kpi-icon">
                <AlertTriangle size={21} />
              </div>

              <div>
                <span>Medium Risk</span>
                <strong>
                  {result.medium_risk}
                </strong>
              </div>

            </div>

            <div className="batch-kpi low">

              <div className="batch-kpi-icon">
                <ShieldCheck size={21} />
              </div>

              <div>
                <span>Low Risk</span>
                <strong>
                  {result.low_risk}
                </strong>
              </div>

            </div>

          </div>

          {/* TABLE */}

          <div className="panel batch-table-panel">

            <div className="panel-header">

              <div>
                <h2>
                  Shipment Risk Results
                </h2>

                <span>
                  Individual model predictions
                </span>
              </div>

            </div>

            <div className="batch-table-wrapper">

              <table className="batch-table">

                <thead>

                  <tr>
                    <th>#</th>
                    <th>Model</th>
                    <th>Risk Type</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Risk</th>
                  </tr>

                </thead>

                <tbody>

                  {result.results.map(
                    (row) => (

                      <tr key={row.row_number}>

                        <td>
                          {row.row_number}
                        </td>

                        <td>
                          {row.model || "—"}
                        </td>

                        <td>
                          {row.risk_type || "—"}
                        </td>

                        <td>
                          {row.prediction === null
                            ? "—"
                            : row.risk_type === "Supply Chain Disruption"
                              ? row.prediction === 1
                                ? "Disruption"
                                : "No Disruption"
                              : row.prediction === 1
                                ? "Late Delivery"
                                : "No Late Delivery"}
                        </td>

                        <td>
                          {row.risk_percentage === null
                            ? "—"
                            : `${row.risk_percentage}%`}
                        </td>

                        <td>

                          <span
                            className={`risk-badge ${
                              row.risk_level
                                .toLowerCase()
                            }`}
                          >
                            {row.risk_level}
                          </span>

                        </td>

                      </tr>

                    )
                  )}

                </tbody>

              </table>

            </div>

          </div>

        </>

      )}

    </div>
  );
}