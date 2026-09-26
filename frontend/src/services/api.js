import axios from "axios";

const API = axios.create({
  baseURL: "http://127.0.0.1:8000/api",
  headers: {
    "Content-Type": "application/json",
  },
});

export const checkHealth = async () => {
  const response = await API.get("/health");
  return response.data;
};

export const getModelInfo = async () => {
  const response = await API.get("/model");
  return response.data;
};

export const getDashboard = async () => {
  const response = await API.get("/dashboard");
  return response.data;
};

export const getAnalytics = async () => {
  const response = await API.get("/analytics");
  return response.data;
};

export const batchPredict = async (file) => {
  const formData = new FormData();

  formData.append("file", file);

  const response = await API.post(
    "/batch-predict",
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );

  return response.data;
};

export const getBatchTemplate = async () => {
  const response = await API.get(
    "/batch-template"
  );

  return response.data;
};

export const predictRisk = async (data) => {
  const response = await API.post("/predict", {
    data,
  });

  return response.data;
};

export const getShipments = async (params = {}) => {
  const response = await API.get("/shipments", {
    params,
  });

  return response.data;
};

export const getShipmentFilters = async () => {
  const response = await API.get(
    "/shipment-filters"
  );

  return response.data;
};

export const getModelIntelligence = async () => {
  const response = await API.get(
    "/model-intelligence"
  );

  return response.data;
};

export const whatIfPrediction = async (data) => {
  const response = await API.post(
    "/what-if",
    {
      data,
    }
  );

  return response.data;
};

export const getPredictionHistory = async (limit = 100) => {
  const response = await API.get(
    "/prediction-history",
    {
      params: { limit },
    }
  );

  return response.data;
};


export const clearPredictionHistory = async () => {
  const response = await API.delete(
    "/prediction-history"
  );

  return response.data;
};

export const getHistorySummary = async () => {
  const response = await API.get("/history-summary");
  return response.data;
};

export default API;