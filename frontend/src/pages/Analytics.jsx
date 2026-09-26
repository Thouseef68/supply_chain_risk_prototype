import { useEffect, useState } from "react";
import {
  BarChart3,
  TrendingUp,
  Globe2,
  Activity,
  AlertTriangle,
} from "lucide-react";

import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  LineChart,
  Line,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

import { getAnalytics } from "../services/api";

import InfoBanner from "../components/InfoBanner";

export default function Analytics() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const axisStyle = {
    fill: "#7f8799",
    fontSize: 11,
  };

  const tooltipStyle = {
    backgroundColor: "#111827",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: "10px",
    color: "#f3f4f6",
  };

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    try {
      setLoading(true);

      const result = await getAnalytics();

      if (!result.success) {
        throw new Error(result.error || "Failed to load analytics");
      }

      setData(result);
    } catch (err) {
      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to load analytics"
      );
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="loading-screen">
        <div className="loading-spinner"></div>
        <p>Loading risk analytics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="analytics-error">
        <AlertTriangle size={22} />
        <div>
          <h3>Analytics unavailable</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  const regionData = [...data.region_risk]
    .sort((a, b) => b.risk_percentage - a.risk_percentage);

  const typeData = [...data.type_risk]
    .sort((a, b) => b.risk_percentage - a.risk_percentage);

  const monthData = [...data.monthly_risk].sort(
    (a, b) => a.month - b.month
  );

  const pieData = typeData.map((item) => ({
    name: item.type,
    value: item.risk_percentage,
  }));

  return (
    <div className="analytics-page">

      <div className="page-header">
        <div>
          <div className="page-kicker">
            <BarChart3 size={16} />
            RISK INTELLIGENCE
          </div>

          <h1>Risk Analytics</h1>

          <p>
            Explore late-delivery risk patterns across operational,
            geographic, and transactional dimensions.
          </p>
        </div>

        <button
          className="secondary-button"
          onClick={loadAnalytics}
        >
          Refresh Analytics
        </button>
      </div>

      <InfoBanner>
        <strong>
          What you're seeing:
        </strong>{" "}
        These charts summarize historical shipment
        patterns and risk levels across regions,
        shipment types and time periods.
      </InfoBanner>

      {/* REGION + TYPE */}

      <div className="analytics-grid two-columns">

        <div className="panel analytics-panel">

          <div className="panel-header">
            <div>
              <h2>Risk by Region</h2>
              <span>
                Late-delivery risk percentage
              </span>
            </div>

            <Globe2 size={20} />
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height={330}>
              <BarChart
                data={regionData}
                layout="vertical"
                margin={{
                  top: 10,
                  right: 20,
                  left: 20,
                  bottom: 10,
                }}
              >
                <CartesianGrid
                  strokeDasharray="3 3"
                  opacity={0.12}
                />

                <XAxis
                  type="number"
                  domain={[0, 100]}
                  tickFormatter={(value) => `${value}%`}
                />

                <YAxis
                  type="category"
                  dataKey="region"
                  width={130}
                  tick={axisStyle}
                />

                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value) => [
                    `${value}%`,
                    "Risk",
                  ]}
                />

                <Bar
                  dataKey="risk_percentage"
                  fill="#3b82f6"
                  radius={[0, 5, 5, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

        </div>

        <div className="panel analytics-panel">

          <div className="panel-header">
            <div>
              <h2>Risk by Transaction Type</h2>
              <span>
                Transaction category risk distribution
              </span>
            </div>

            <Activity size={20} />
          </div>

          <div className="chart-container">
            <ResponsiveContainer width="100%" height={330}>
              <BarChart data={typeData}>
                <CartesianGrid
                  strokeDasharray="3 3"
                  opacity={0.12}
                />

                <XAxis
                  dataKey="type"
                  tick={axisStyle}
                />

                <YAxis
                  domain={[0, 100]}
                  tickFormatter={(value) => `${value}%`}
                />

                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value) => [
                    `${value}%`,
                    "Risk",
                  ]}
                />

                <Bar
                  dataKey="risk_percentage"
                  fill="#8b5cf6"
                  radius={[5, 5, 0, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>

        </div>

      </div>

      {/* MONTHLY TREND */}

      <div className="panel analytics-panel">

        <div className="panel-header">

          <div>
            <h2>Monthly Risk Trend</h2>
            <span>
              Variation in late-delivery risk by order month
            </span>
          </div>

          <TrendingUp size={20} />

        </div>

        <div className="chart-container large-chart">

          <ResponsiveContainer width="100%" height={350}>

            <LineChart
              data={monthData}
              margin={{
                top: 20,
                right: 30,
                left: 10,
                bottom: 10,
              }}
            >

              <CartesianGrid
                strokeDasharray="3 3"
                opacity={0.12}
              />

              <XAxis
                dataKey="month"
                tickFormatter={(value) => `Month ${value}`}
              />

              <YAxis
                domain={[0, 100]}
                tickFormatter={(value) => `${value}%`}
              />

              <Tooltip
                contentStyle={tooltipStyle}
                labelFormatter={(value) => `Month ${value}`}
                formatter={(value) => [
                  `${value}%`,
                  "Risk",
                ]}
              />

              <Line
                type="monotone"
                dataKey="risk_percentage"
                stroke="#22c55e"
                strokeWidth={3}
                dot={{ r: 4, fill: "#22c55e" }}
                activeDot={{ r: 7 }}
              />

            </LineChart>

          </ResponsiveContainer>

        </div>

      </div>

      {/* TRANSACTION PIE */}

      <div className="analytics-grid two-columns">

        <div className="panel analytics-panel">

          <div className="panel-header">

            <div>
              <h2>Transaction Risk Profile</h2>
              <span>
                Relative risk across transaction types
              </span>
            </div>

            <BarChart3 size={20} />

          </div>

          <div className="chart-container">

            <ResponsiveContainer width="100%" height={330}>

              <PieChart>

                <Pie
                  data={pieData}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={110}
                  label={({ name, value }) =>
                    `${name}: ${value}%`
                  }
                >

                  {pieData.map((_, index) => {
              const colors = [
                "#3b82f6",
                "#8b5cf6",
                "#22c55e",
                "#f59e0b",
                "#ef4444",
              ];

              return (
                <Cell
                  key={index}
                  fill={colors[index % colors.length]}
                />
              );
            })}

                </Pie>

                <Tooltip
                  contentStyle={tooltipStyle}
                  formatter={(value) => [
                    `${value}%`,
                    "Risk",
                  ]}
                />

                <Legend />

              </PieChart>

            </ResponsiveContainer>

          </div>

        </div>

        {/* REGION TABLE */}

        <div className="panel analytics-panel">

          <div className="panel-header">

            <div>
              <h2>Regional Risk Overview</h2>
              <span>
                Orders and observed risk rate
              </span>
            </div>

            <AlertTriangle size={20} />

          </div>

          <div className="risk-table">

            <div className="risk-table-header">
              <span>Region</span>
              <span>Orders</span>
              <span>Risk</span>
            </div>

            {regionData.map((item) => (

              <div
                className="risk-table-row"
                key={item.region}
              >

                <span>
                  {item.region}
                </span>

                <span>
                  {item.orders.toLocaleString()}
                </span>

                <span className="risk-value">
                  {item.risk_percentage}%
                </span>

              </div>

            ))}

          </div>

        </div>

      </div>

      {/* HEATMAP */}

      <div className="panel analytics-panel">

        <div className="panel-header">

          <div>
            <h2>Market × Region Risk Matrix</h2>
            <span>
              Observed late-delivery risk across market and
              regional combinations
            </span>
          </div>

          <Globe2 size={20} />

        </div>

        <div className="heatmap-container">

          {data.heatmap.map((item, index) => {

            const intensity =
              Math.min(item.risk_percentage, 100);

            return (
              <div
                className="heatmap-cell"
                key={`${item.market}-${item.region}-${index}`}
                style={{
                  opacity:
                    0.35 + intensity / 180,
                }}
              >

                <div className="heatmap-market">
                  {item.market}
                </div>

                <div className="heatmap-region">
                  {item.region}
                </div>

                <strong>
                  {item.risk_percentage}%
                </strong>

              </div>
            );
          })}

        </div>

      </div>

    </div>
  );
}