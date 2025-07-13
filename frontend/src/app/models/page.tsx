"use client";
import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from "recharts";
import toast, { Toaster } from "react-hot-toast";
import { API_ENDPOINTS } from '../../config/api';

interface ModelInfo {
  id: string;
  name: string;
  version: string;
  model_type: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1_score: number;
  last_updated: string;
  status: string;
}

export default function ModelsPage() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retrainStatus, setRetrainStatus] = useState<string | null>(null);

  useEffect(() => {
    fetch(API_ENDPOINTS.MODELS)
      .then(res => res.json())
      .then(data => {
        if (Array.isArray(data)) {
          setModels(data);
        } else if (data && Array.isArray(data.models)) {
          setModels(data.models);
        } else {
          setModels([]);
          setError("Unexpected response from models API");
          console.error("Unexpected models API response:", data);
        }
      })
      .catch(() => setError("Failed to load models"))
      .finally(() => setLoading(false));
  }, []);

  const handleRetrain = async (modelType: string) => {
    setRetrainStatus("Starting retraining...");
    toast.loading("Retraining started...");
    try {
      const res = await fetch(API_ENDPOINTS.MODELS_TRAIN, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model_type: modelType }),
      });
      if (!res.ok) throw new Error("Failed to start retraining");
      const data = await res.json();
      setRetrainStatus(`Retraining started (job: ${data.job_id})`);
      toast.success("Retraining started!");
    } catch {
      setRetrainStatus("Failed to start retraining");
      toast.error("Failed to start retraining");
    }
  };

  // Prepare data for the chart
  const chartData = models.map(model => ({
    name: model.name,
    Accuracy: model.accuracy,
    "F1 Score": model.f1_score,
  }));

  return (
    <div className="max-w-3xl mx-auto py-10 px-4">
      <Toaster position="top-right" />
      <h1 className="text-2xl font-bold mb-6">Model Dashboard</h1>
      {loading && <div>Loading models...</div>}
      {error && <div className="text-red-600">{error}</div>}
      {retrainStatus && <div className="mb-4 text-blue-700">{retrainStatus}</div>}
      {models.length > 0 && (
        <div className="mb-8 bg-white dark:bg-gray-900 p-4 rounded shadow">
          <h2 className="font-semibold mb-2">Model Performance</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} margin={{ top: 20, right: 30, left: 0, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Legend />
              <Bar dataKey="Accuracy" fill="#2563eb" />
              <Bar dataKey="F1 Score" fill="#10b981" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      <table className="w-full border mt-4 bg-white dark:bg-gray-900 rounded shadow" aria-label="Model table">
        <thead>
          <tr className="bg-blue-100 dark:bg-blue-900">
            <th className="p-2">Name</th>
            <th className="p-2">Type</th>
            <th className="p-2">Version</th>
            <th className="p-2">Accuracy</th>
            <th className="p-2">F1</th>
            <th className="p-2">Status</th>
            <th className="p-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {models.map(model => (
            <tr key={model.id} className="border-t">
              <td className="p-2 font-semibold">{model.name}</td>
              <td className="p-2">{model.model_type}</td>
              <td className="p-2">{model.version}</td>
              <td className="p-2">{model.accuracy}</td>
              <td className="p-2">{model.f1_score}</td>
              <td className="p-2">{model.status}</td>
              <td className="p-2">
                <button
                  className="bg-blue-600 text-white px-3 py-1 rounded hover:bg-blue-700 text-sm"
                  onClick={() => handleRetrain(model.model_type)}
                  aria-label={`Retrain ${model.name}`}
                >
                  Retrain
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
} 