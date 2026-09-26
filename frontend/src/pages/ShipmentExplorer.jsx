import { useEffect, useState } from "react";
import {
  Search,
  Database,
  ChevronLeft,
  ChevronRight,
  MapPin,
  Package,
  Truck,
  X,
} from "lucide-react";

import {
  getShipments,
  getShipmentFilters,
} from "../services/api";

import InfoBanner from "../components/InfoBanner";

export default function ShipmentExplorer() {

  const [shipments, setShipments] = useState([]);
  const [filters, setFilters] = useState({
    shipping_modes: [],
    markets: [],
  });

  const [search, setSearch] = useState("");
  const [risk, setRisk] = useState("ALL");
  const [shippingMode, setShippingMode] =
    useState("ALL");
  const [market, setMarket] =
    useState("ALL");

  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);
  const [totalPages, setTotalPages] = useState(0);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [selectedShipment, setSelectedShipment] =
    useState(null);

  useEffect(() => {
    loadFilters();
  }, []);

  useEffect(() => {
    loadShipments();
  }, [
    page,
    risk,
    shippingMode,
    market,
  ]);

  const loadFilters = async () => {
    try {
      const result =
        await getShipmentFilters();

      if (result.success) {
        setFilters(result);
      }
    } catch (err) {
      console.error(err);
    }
  };

  const loadShipments = async () => {

    try {

      setLoading(true);
      setError("");

      const result =
        await getShipments({
          search,
          risk,
          shipping_mode: shippingMode,
          market,
          page,
          limit: 25,
        });

      if (!result.success) {
        throw new Error(result.error);
      }

      setShipments(result.shipments);
      setTotal(result.total);
      setTotalPages(result.total_pages);

    } catch (err) {

      setError(
        err.message ||
        "Unable to load shipments."
      );

    } finally {

      setLoading(false);

    }
  };

  const handleSearch = (event) => {

    if (event.key === "Enter") {
      setPage(1);
      loadShipments();
    }
  };

  const clearSearch = () => {
    setSearch("");
    setPage(1);

    setTimeout(() => {
      loadShipments();
    }, 0);
  };

  const resetFilters = () => {
    setSearch("");
    setRisk("ALL");
    setShippingMode("ALL");
    setMarket("ALL");
    setPage(1);
  };

  return (
    <div className="explorer-page">

      {/* HEADER */}

      <div className="page-header">

        <div>

          <div className="page-kicker">
            <Database size={16} />
            OPERATIONAL DATA
          </div>

          <h1>Shipment Explorer</h1>

          <p>
            Search and investigate shipment records
            across the DataCo supply chain dataset.
          </p>

        </div>

        <div className="explorer-count">
          <strong>
            {total.toLocaleString()}
          </strong>

          <span>matching shipments</span>
        </div>

      </div>

      <InfoBanner>
        <strong>
          How to use:
        </strong>{" "}
        Search or filter shipments to inspect
        individual records and their associated
        delivery-risk information.
      </InfoBanner>

      {/* SEARCH */}

      <div className="panel explorer-controls">

        <div className="search-box">

          <Search size={18} />

          <input
            type="text"
            placeholder="Search city, country, product, state or region..."
            value={search}
            onChange={(event) =>
              setSearch(event.target.value)
            }
            onKeyDown={handleSearch}
          />

          {search && (
            <button
              onClick={clearSearch}
              className="clear-search"
            >
              <X size={16} />
            </button>
          )}

        </div>

        <select
          value={risk}
          onChange={(event) => {
            setRisk(event.target.value);
            setPage(1);
          }}
        >
          <option value="ALL">
            All Risk Levels
          </option>

          <option value="HIGH">
            High Risk
          </option>

          <option value="LOW">
            Low Risk
          </option>
        </select>

        <select
          value={shippingMode}
          onChange={(event) => {
            setShippingMode(event.target.value);
            setPage(1);
          }}
        >
          <option value="ALL">
            All Shipping Modes
          </option>

          {filters.shipping_modes.map(
            (mode) => (
              <option
                value={mode}
                key={mode}
              >
                {mode}
              </option>
            )
          )}

        </select>

        <select
          value={market}
          onChange={(event) => {
            setMarket(event.target.value);
            setPage(1);
          }}
        >

          <option value="ALL">
            All Markets
          </option>

          {filters.markets.map(
            (item) => (
              <option
                value={item}
                key={item}
              >
                {item}
              </option>
            )
          )}

        </select>

        <button
          className="secondary-button"
          onClick={resetFilters}
        >
          Reset
        </button>

      </div>

      {/* ERROR */}

      {error && (
        <div className="explorer-error">
          {error}
        </div>
      )}

      {/* TABLE */}

      <div className="panel explorer-table-panel">

        <div className="panel-header">

          <div>

            <h2>
              Shipment Records
            </h2>

            <span>
              Showing page {page} of{" "}
              {totalPages || 1}
            </span>

          </div>

        </div>

        {loading ? (

          <div className="explorer-loading">
            Loading shipment records...
          </div>

        ) : (

          <div className="explorer-table-wrapper">

            <table className="explorer-table">

              <thead>

                <tr>

                  <th>Order ID</th>

                  <th>Route</th>

                  <th>Market</th>

                  <th>Shipping</th>

                  <th>Scheduled</th>

                  <th>Sales</th>

                  <th>Risk</th>

                </tr>

              </thead>

              <tbody>

                {shipments.map(
                  (shipment) => (

                    <tr
                      key={shipment.id}
                      onClick={() =>
                        setSelectedShipment(
                          shipment
                        )
                      }
                    >

                      <td>
                        <strong>
                          #{shipment.id}
                        </strong>
                      </td>

                      <td>

                        <div className="route-cell">

                          <MapPin size={15} />

                          <div>

                            <strong>
                              {shipment.order_city}
                            </strong>

                            <span>
                              {shipment.order_country}
                            </span>

                          </div>

                        </div>

                      </td>

                      <td>
                        {shipment.market}
                      </td>

                      <td>

                        <div className="shipping-cell">

                          <Truck size={14} />

                          {shipment.shipping_mode}

                        </div>

                      </td>

                      <td>
                        {shipment.scheduled_days}
                        {" "}days
                      </td>

                      <td>
                        $
                        {shipment.sales?.toLocaleString(
                          undefined,
                          {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2,
                          }
                        )}
                      </td>

                      <td>

                        <span
                          className={`risk-badge ${
                            shipment.risk.toLowerCase()
                          }`}
                        >
                          {shipment.risk}
                        </span>

                      </td>

                    </tr>

                  )
                )}

                {shipments.length === 0 && (
                  <tr>

                    <td
                      colSpan="7"
                      className="empty-state"
                    >
                      No shipments match your
                      current filters.
                    </td>

                  </tr>
                )}

              </tbody>

            </table>

          </div>

        )}

        {/* PAGINATION */}

        <div className="explorer-pagination">

          <span>
            {total === 0
              ? "0"
              : `${((page - 1) * 25) + 1}-${Math.min(
                  page * 25,
                  total
                )}`}
            {" "}of {total.toLocaleString()}
          </span>

          <div>

            <button
              disabled={page <= 1}
              onClick={() =>
                setPage((value) =>
                  Math.max(1, value - 1)
                )
              }
            >
              <ChevronLeft size={17} />
            </button>

            <span>
              Page {page}
            </span>

            <button
              disabled={
                page >= totalPages
              }
              onClick={() =>
                setPage((value) =>
                  Math.min(
                    totalPages,
                    value + 1
                  )
                )
              }
            >
              <ChevronRight size={17} />
            </button>

          </div>

        </div>

      </div>

      {/* DETAIL PANEL */}

      {selectedShipment && (

        <div
          className="shipment-overlay"
          onClick={() =>
            setSelectedShipment(null)
          }
        >

          <div
            className="shipment-detail"
            onClick={(event) =>
              event.stopPropagation()
            }
          >

            <div className="shipment-detail-header">

              <div>

                <span>
                  SHIPMENT DETAILS
                </span>

                <h2>
                  #{selectedShipment.id}
                </h2>

              </div>

              <button
                onClick={() =>
                  setSelectedShipment(null)
                }
              >
                <X size={20} />
              </button>

            </div>

            <div className="shipment-risk-banner">

              <div>

                <span>
                  LATE DELIVERY RISK
                </span>

                <strong>
                  {selectedShipment.risk}
                </strong>

              </div>

              <Package size={30} />

            </div>

            <div className="shipment-detail-grid">

              <div>
                <span>Order City</span>
                <strong>
                  {selectedShipment.order_city}
                </strong>
              </div>

              <div>
                <span>Country</span>
                <strong>
                  {selectedShipment.order_country}
                </strong>
              </div>

              <div>
                <span>State</span>
                <strong>
                  {selectedShipment.order_state}
                </strong>
              </div>

              <div>
                <span>Market</span>
                <strong>
                  {selectedShipment.market}
                </strong>
              </div>

              <div>
                <span>Region</span>
                <strong>
                  {selectedShipment.region}
                </strong>
              </div>

              <div>
                <span>Shipping Mode</span>
                <strong>
                  {selectedShipment.shipping_mode}
                </strong>
              </div>

              <div>
                <span>Scheduled Delivery</span>
                <strong>
                  {selectedShipment.scheduled_days}
                  {" "}days
                </strong>
              </div>

              <div>
                <span>Quantity</span>
                <strong>
                  {selectedShipment.quantity}
                </strong>
              </div>

              <div>
                <span>Sales</span>
                <strong>
                  $
                  {selectedShipment.sales?.toFixed(2)}
                </strong>
              </div>

              <div>
                <span>Customer City</span>
                <strong>
                  {selectedShipment.customer_city}
                </strong>
              </div>

            </div>

          </div>

        </div>

      )}

    </div>
  );
}